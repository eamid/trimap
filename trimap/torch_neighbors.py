"""GPU-friendly nearest-neighbor backends for the PyTorch TriMap implementation.

The native backend is an exact, blocked search which never materializes the full
``n x n`` distance matrix.  Faiss and NVIDIA cuVS are optional accelerators.  All
backends return the query point itself in column zero, followed by neighbors in
ascending distance order.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from typing import Literal

import torch
import torch.nn.functional as F

Metric = Literal["euclidean", "manhattan", "cosine", "angular", "hamming", "chebyshev"]
KNNBackend = Literal["auto", "torch", "faiss", "faiss-flat", "faiss-ivf", "cuvs-cagra"]


def canonical_metric(metric: str) -> str:
    """Return the canonical spelling for a supported distance metric."""
    metric = metric.lower()
    if metric == "angular":
        return "cosine"
    supported = {"euclidean", "manhattan", "cosine", "hamming", "chebyshev"}
    if metric not in supported:
        raise ValueError(
            f"Unsupported distance {metric!r}; expected one of {sorted(supported)}."
        )
    return metric


def rowwise_distances(x1: torch.Tensor, x2: torch.Tensor, metric: str) -> torch.Tensor:
    """Compute distances between corresponding rows of two tensors."""
    if x1.shape != x2.shape:
        raise ValueError(
            f"rowwise inputs must have equal shapes, got {x1.shape} and {x2.shape}"
        )
    metric = canonical_metric(metric)
    delta = x1 - x2
    if metric == "euclidean":
        return torch.linalg.vector_norm(delta, dim=-1)
    if metric == "manhattan":
        return delta.abs().sum(dim=-1)
    if metric == "cosine":
        return 1.0 - F.cosine_similarity(x1, x2, dim=-1, eps=1e-20)
    if metric == "hamming":
        return (x1 != x2).sum(dim=-1, dtype=x1.dtype)
    return delta.abs().amax(dim=-1)


def _pairwise_block(xq: torch.Tensor, xb: torch.Tensor, metric: str) -> torch.Tensor:
    metric = canonical_metric(metric)
    if metric == "cosine":
        return 1.0 - xq @ xb.T
    p = {"euclidean": 2.0, "manhattan": 1.0, "hamming": 0.0, "chebyshev": float("inf")}[
        metric
    ]
    return torch.cdist(xq, xb, p=p)


@torch.no_grad()
def torch_knn(
    inputs: torch.Tensor,
    k: int,
    *,
    metric: str = "euclidean",
    query_batch_size: int = 4096,
    database_batch_size: int = 16384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact blocked k-NN search using only PyTorch operations.

    The two batch sizes independently bound the temporary pairwise-distance
    matrix.  This makes the implementation useful on CUDA, MPS, and CPU while
    retaining exact results.
    """
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be two-dimensional, got shape {inputs.shape}")
    n = inputs.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    if query_batch_size < 1 or database_batch_size < 1:
        raise ValueError("batch sizes must be positive")

    metric = canonical_metric(metric)
    search_inputs = (
        F.normalize(inputs, dim=1, eps=1e-20) if metric == "cosine" else inputs
    )
    all_distances: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []

    for q_start in range(0, n, query_batch_size):
        q_end = min(q_start + query_batch_size, n)
        queries = search_inputs[q_start:q_end]
        best_distances = torch.empty(
            (q_end - q_start, 0), dtype=inputs.dtype, device=inputs.device
        )
        best_indices = torch.empty(
            (q_end - q_start, 0), dtype=torch.long, device=inputs.device
        )

        for b_start in range(0, n, database_batch_size):
            b_end = min(b_start + database_batch_size, n)
            distances = _pairwise_block(queries, search_inputs[b_start:b_end], metric)

            # Give self strict priority while selecting.  This matters for
            # duplicate observations (especially Hamming data), where an
            # arbitrary equal-distance point could otherwise occupy column 0.
            overlap_start = max(q_start, b_start)
            overlap_end = min(q_end, b_end)
            if overlap_start < overlap_end:
                diagonal = torch.arange(
                    overlap_start, overlap_end, device=inputs.device
                )
                distances[diagonal - q_start, diagonal - b_start] = -float("inf")

            block_indices = torch.arange(b_start, b_end, device=inputs.device).expand(
                q_end - q_start, -1
            )
            candidate_distances = torch.cat((best_distances, distances), dim=1)
            candidate_indices = torch.cat((best_indices, block_indices), dim=1)
            keep = min(k, candidate_distances.shape[1])
            best_distances, positions = torch.topk(
                candidate_distances, keep, dim=1, largest=False, sorted=True
            )
            best_indices = torch.gather(candidate_indices, 1, positions)

        best_distances[:, 0] = 0.0
        all_distances.append(best_distances)
        all_indices.append(best_indices)

    return torch.cat(all_indices), torch.cat(all_distances)


def _ensure_self_first(
    indices: torch.Tensor,
    distances: torch.Tensor,
    k: int,
    *,
    row_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize a self-query result even when an ANN omitted the query."""
    n = indices.shape[0]
    rows = torch.arange(row_offset, row_offset + n, device=indices.device)[:, None]
    if k == 1:
        return rows, torch.zeros((n, 1), dtype=distances.dtype, device=distances.device)

    nonself_distances = distances.masked_fill(indices == rows, float("inf"))
    keep = min(k - 1, nonself_distances.shape[1])
    nearest_distances, positions = torch.topk(
        nonself_distances, keep, dim=1, largest=False, sorted=True
    )
    nearest_indices = torch.gather(indices, 1, positions)
    if keep != k - 1 or torch.isinf(nearest_distances).any():
        raise RuntimeError(
            "nearest-neighbor backend returned too few distinct non-self points"
        )
    return (
        torch.cat((rows, nearest_indices), dim=1),
        torch.cat(
            (
                torch.zeros((n, 1), dtype=distances.dtype, device=distances.device),
                nearest_distances,
            ),
            dim=1,
        ),
    )


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve_knn_backend(
    backend: str,
    inputs: torch.Tensor,
    metric: str,
    *,
    ann_threshold: int = 50_000,
) -> str:
    """Resolve ``auto`` to an installed backend appropriate for the input."""
    backend = backend.lower()
    metric = canonical_metric(metric)
    valid = {"auto", "torch", "faiss", "faiss-flat", "faiss-ivf", "cuvs-cagra"}
    if backend not in valid:
        raise ValueError(
            f"Unknown knn_backend {backend!r}; expected one of {sorted(valid)}"
        )
    if backend != "auto":
        return "faiss-flat" if backend == "faiss" else backend

    accelerated_metric = metric in {"euclidean", "cosine"}
    if inputs.is_cuda and accelerated_metric:
        if inputs.shape[0] >= ann_threshold and _has_module("cuvs"):
            return "cuvs-cagra"
        if _has_module("faiss"):
            return "faiss-ivf" if inputs.shape[0] >= ann_threshold else "faiss-flat"
    if (
        inputs.device.type == "cpu"
        and sys.platform != "darwin"
        and accelerated_metric
        and _has_module("faiss")
    ):
        return "faiss-ivf" if inputs.shape[0] >= ann_threshold else "faiss-flat"
    return "torch"


@torch.no_grad()
def _faiss_knn(
    inputs: torch.Tensor,
    k: int,
    *,
    metric: str,
    approximate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    metric = canonical_metric(metric)
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("Faiss backends support only euclidean and cosine distances")
    if inputs.device.type == "cpu" and sys.platform == "darwin":
        raise RuntimeError(
            "CPU Faiss and PyTorch commonly load conflicting OpenMP runtimes on macOS. "
            "Use knn_backend='torch'; CUDA Faiss remains supported on Linux."
        )
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Faiss was requested but is not installed. Install faiss-cpu, or a "
            "CUDA-enabled Faiss build for zero-copy GPU search."
        ) from exc

    x = inputs.detach().to(dtype=torch.float32).contiguous()
    if metric == "cosine":
        x = F.normalize(x, dim=1, eps=1e-20)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
        quantizer = faiss.IndexFlatIP(x.shape[1])
    else:
        faiss_metric = faiss.METRIC_L2
        quantizer = faiss.IndexFlatL2(x.shape[1])

    if approximate:
        # Faiss recommends enough training points per inverted list.  The cap
        # avoids a poor/invalid IVF training regime on medium-sized datasets.
        nlist = max(1, min(int(math.sqrt(x.shape[0])), x.shape[0] // 39))
        cpu_index = faiss.IndexIVFFlat(quantizer, x.shape[1], nlist, faiss_metric)
        cpu_index.nprobe = min(nlist, max(8, int(math.sqrt(nlist))))
    else:
        cpu_index = quantizer

    index = cpu_index
    resources = None
    if x.is_cuda:
        # This patches the Faiss Python bindings for zero-copy torch tensors.
        # Keep the CPU path on NumPy: apart from avoiding needless binding
        # machinery, some Faiss/PyTorch version pairs have unsafe CPU wrappers.
        import faiss.contrib.torch_utils  # type: ignore

        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "The installed Faiss build has no GPU support; use knn_backend='torch' "
                "or install a CUDA-enabled Faiss build."
            )
        resources = faiss.StandardGpuResources()
        resources.setDefaultNullStreamAllDevices()
        index = faiss.index_cpu_to_gpu(resources, x.device.index or 0, cpu_index)

    search_inputs = x if x.is_cuda else x.cpu().numpy()
    if approximate:
        index.train(search_inputs)
    index.add(search_inputs)
    distances, indices = index.search(search_inputs, min(x.shape[0], k + 1))
    indices = torch.as_tensor(indices, device=inputs.device, dtype=torch.long)
    distances = torch.as_tensor(distances, device=inputs.device, dtype=inputs.dtype)
    if metric == "euclidean":
        distances = distances.clamp_min_(0).sqrt_()
    else:
        distances = 1.0 - distances
    return _ensure_self_first(indices, distances, k)


@torch.no_grad()
def _cuvs_cagra_knn(
    inputs: torch.Tensor,
    k: int,
    *,
    metric: str,
    query_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not inputs.is_cuda:
        raise ValueError("cuVS CAGRA requires a CUDA tensor")
    metric = canonical_metric(metric)
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("cuVS CAGRA supports only euclidean and cosine distances here")
    try:
        from cuvs.neighbors import cagra  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "cuVS was requested but is not installed. Install the cuvs-cu12 or "
            "cuvs-cu13 wheel matching the CUDA runtime."
        ) from exc

    x = inputs.detach().to(dtype=torch.float32).contiguous()
    if metric == "cosine":
        x = F.normalize(x, dim=1, eps=1e-20)
    build_options = {
        "metric": "sqeuclidean",
        "intermediate_graph_degree": 128,
        "graph_degree": 64,
    }
    try:
        # NN-descent is both faster and robust to highly duplicated data such
        # as KDDCup99, for which cuVS's default IVF-PQ builder can produce an
        # invalid intermediate graph containing duplicate neighbor nodes.
        build_params = cagra.IndexParams(**build_options, build_algo="nn_descent")
    except TypeError:
        # cuVS releases before the NN-descent option was exposed still work
        # with their default graph builder.
        build_params = cagra.IndexParams(**build_options)
    index = cagra.build(build_params, x)

    # CAGRA's finished graph already contains ``graph_degree`` candidate
    # neighbors for every indexed point.  Since this is always a self-query,
    # searching every input against the graph again is redundant and becomes
    # dominant at multi-million-point scale.  Rank the graph candidates by
    # their actual distance in bounded batches instead.
    graph = index.graph
    if graph.shape[1] < k - 1:
        raise RuntimeError(
            f"CAGRA graph has only {graph.shape[1]} neighbors; {k - 1} required"
        )
    distance_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    for start in range(0, x.shape[0], query_batch_size):
        end = min(start + query_batch_size, x.shape[0])
        # cuVS exposes an immutable CUDA array view for the graph, while
        # PyTorch rejects read-only ``__cuda_array_interface__`` objects.
        # Copy only this bounded slice through host memory before returning it
        # to the device; the large feature and graph storage remains in CUDA.
        graph_slice = graph.slice_rows(start, end).copy_to_host()
        candidate_indices = torch.as_tensor(
            graph_slice, device=inputs.device, dtype=torch.long
        )
        candidate_vectors = x[candidate_indices]
        if metric == "euclidean":
            candidate_vectors.sub_(x[start:end, None, :]).square_()
            candidate_distances = candidate_vectors.sum(dim=-1).sqrt_()
        else:
            candidate_vectors.mul_(x[start:end, None, :])
            candidate_distances = 1.0 - candidate_vectors.sum(dim=-1)
        indices, distances = _ensure_self_first(
            candidate_indices,
            candidate_distances,
            k,
            row_offset=start,
        )
        distance_parts.append(distances)
        index_parts.append(indices)
    return torch.cat(index_parts), torch.cat(distance_parts).to(inputs.dtype)


@torch.no_grad()
def knn_search(
    inputs: torch.Tensor,
    k: int,
    *,
    metric: str = "euclidean",
    backend: str = "auto",
    query_batch_size: int = 4096,
    database_batch_size: int = 16384,
    ann_threshold: int = 50_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find self k-NN with a native, Faiss, or cuVS backend.

    ``auto`` prefers cuVS CAGRA for large CUDA inputs, then Faiss, and always
    falls back to the exact blocked PyTorch implementation if an optional
    accelerator is unavailable or incompatible.
    """
    resolved = resolve_knn_backend(backend, inputs, metric, ann_threshold=ann_threshold)
    try:
        if resolved == "torch":
            return torch_knn(
                inputs,
                k,
                metric=metric,
                query_batch_size=query_batch_size,
                database_batch_size=database_batch_size,
            )
        if resolved in {"faiss-flat", "faiss-ivf"}:
            return _faiss_knn(
                inputs, k, metric=metric, approximate=resolved == "faiss-ivf"
            )
        return _cuvs_cagra_knn(
            inputs, k, metric=metric, query_batch_size=query_batch_size
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        if backend != "auto":
            raise
        warnings.warn(
            f"Automatic k-NN backend {resolved!r} failed ({exc}); falling back to "
            "the exact PyTorch backend.",
            RuntimeWarning,
        )
        return torch_knn(
            inputs,
            k,
            metric=metric,
            query_batch_size=query_batch_size,
            database_batch_size=database_batch_size,
        )


__all__ = [
    "KNNBackend",
    "Metric",
    "canonical_metric",
    "knn_search",
    "resolve_knn_backend",
    "rowwise_distances",
    "torch_knn",
]
