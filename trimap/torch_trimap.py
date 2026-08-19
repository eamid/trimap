"""Tensor-native, GPU-parallel TriMap.

This module mirrors the components of the original NumPy/Numba implementation
with batched PyTorch kernels.  Inputs, neighbors, triplets, weights, gradients,
optimizer state, and the embedding stay on the selected device.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from .torch_neighbors import (
    _ensure_self_first,
    canonical_metric,
    knn_search,
    resolve_knn_backend,
    rowwise_distances,
)

_PCA_DIM = 100
_RAND_WEIGHT_SCALE = 0.1
_INIT_PCA_SCALE = 0.01
_INIT_RANDOM_SCALE = 0.0001
_INIT_MOMENTUM = 0.5
_FINAL_MOMENTUM = 0.8
_SWITCH_ITER = 250
_MIN_GAIN = 0.01
_INCREASE_GAIN = 0.2
_DAMP_GAIN = 0.8
_LEGACY_LEARNING_RATE_SCALE = 0.5
_RETURN_EVERY = 10
_DISPLAY_EVERY = 100


def tempered_log(x: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply the temperature-deformed logarithm used for triplet weights."""
    if abs(temperature - 1.0) < 1e-5:
        return torch.log(x)
    return (x.pow(1.0 - temperature) - 1.0) / (1.0 - temperature)


def _slices(length: int, batch_size: int | None) -> Iterator[slice]:
    if batch_size is None or batch_size >= length:
        yield slice(0, length)
        return
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


def sliced_distances(
    indices1: torch.Tensor,
    indices2: torch.Tensor,
    inputs: torch.Tensor,
    metric: str,
    *,
    batch_size: int | None = 1_000_000,
    pairwise_distances: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute indexed row distances in memory-bounded, parallel slices."""
    if indices1.shape != indices2.shape:
        raise ValueError("distance index arrays must have equal shape")
    if pairwise_distances is not None:
        return pairwise_distances[indices1, indices2]
    result = []
    for part in _slices(indices1.numel(), batch_size):
        result.append(
            rowwise_distances(inputs[indices1[part]], inputs[indices2[part]], metric)
        )
    return torch.cat(result) if len(result) > 1 else result[0]


def _randint(
    high: int,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randint(high, shape, device=device, generator=generator)


def rejection_sample(
    rejects: torch.Tensor,
    n_samples: int,
    maxval: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample unique integers per row while rejecting a row-specific set.

    Each sampling column is generated for every point at once.  Only invalid
    entries are redrawn, so the small control loop launches massively parallel
    kernels rather than iterating over observations in Python.
    """
    if rejects.ndim != 2:
        raise ValueError("rejects must have shape (rows, rejected_values)")
    if n_samples < 1:
        return torch.empty(
            (rejects.shape[0], 0), dtype=torch.long, device=rejects.device
        )
    if maxval <= rejects.shape[1] + n_samples - 1:
        raise ValueError(
            "not enough points to draw distinct outliers after rejected neighbors"
        )
    rejects = rejects.long()
    rows = rejects.shape[0]
    samples = torch.empty((rows, n_samples), dtype=torch.long, device=rejects.device)
    for column in range(n_samples):
        candidate = _randint(
            maxval, (rows,), device=rejects.device, generator=generator
        )
        invalid = (candidate[:, None] == rejects).any(dim=1)
        if column:
            invalid |= (candidate[:, None] == samples[:, :column]).any(dim=1)
        while bool(invalid.any()):
            redraw = _randint(
                maxval, (rows,), device=rejects.device, generator=generator
            )
            candidate = torch.where(invalid, redraw, candidate)
            invalid = (candidate[:, None] == rejects).any(dim=1)
            if column:
                invalid |= (candidate[:, None] == samples[:, :column]).any(dim=1)
        samples[:, column] = candidate
    return samples


def sample_knn_triplets(
    neighbors: torch.Tensor,
    n_inliers: int,
    n_outliers: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample nearest-neighbor triplets in anchor/inlier/outlier order."""
    n = neighbors.shape[0]
    if neighbors.shape[1] < n_inliers + 1:
        raise ValueError("neighbors must contain self plus all requested inliers")
    inlier_neighbors = neighbors[:, 1 : n_inliers + 1]
    anchors = torch.arange(n, device=neighbors.device)[:, None]
    # Legacy TriMap excludes self and the inlier prefix up to the current
    # inlier.  Later inliers remain valid outlier candidates for earlier
    # constraints.  Keep that distribution while drawing every row in
    # parallel on the accelerator.
    outlier_groups = [
        rejection_sample(
            torch.cat((anchors, inlier_neighbors[:, : inlier + 1]), dim=1),
            n_outliers,
            n,
            generator=generator,
        )
        for inlier in range(n_inliers)
    ]
    outliers = torch.stack(outlier_groups, dim=1)
    expanded_anchors = anchors[:, :, None].expand(
        n, n_inliers, n_outliers
    )
    inliers = inlier_neighbors[:, :, None].expand(n, n_inliers, n_outliers)
    return (
        torch.stack((expanded_anchors, inliers, outliers), dim=-1)
        .reshape(-1, 3)
        .long()
    )


def sample_random_triplets(
    inputs: torch.Tensor,
    n_random: int,
    sig: torch.Tensor,
    metric: str,
    *,
    generator: torch.Generator | None = None,
    distance_batch_size: int | None = 1_000_000,
    pairwise_distances: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random ordered triplets and their untransformed weights."""
    n = inputs.shape[0]
    if n_random == 0:
        return (
            torch.empty((0, 3), dtype=torch.long, device=inputs.device),
            torch.empty((0,), dtype=inputs.dtype, device=inputs.device),
        )
    if n < 3:
        raise ValueError("at least three points are required for random triplets")
    anchors = torch.arange(n, device=inputs.device)[:, None].expand(n, n_random)
    similar = _randint(n, (n, n_random), device=inputs.device, generator=generator)
    invalid = similar == anchors
    while bool(invalid.any()):
        redraw = _randint(n, (n, n_random), device=inputs.device, generator=generator)
        similar = torch.where(invalid, redraw, similar)
        invalid = similar == anchors

    outlier = _randint(n, (n, n_random), device=inputs.device, generator=generator)
    invalid = (outlier == anchors) | (outlier == similar)
    while bool(invalid.any()):
        redraw = _randint(n, (n, n_random), device=inputs.device, generator=generator)
        outlier = torch.where(invalid, redraw, outlier)
        invalid = (outlier == anchors) | (outlier == similar)

    anchors = anchors.reshape(-1)
    similar = similar.reshape(-1)
    outlier = outlier.reshape(-1)
    d_sim = sliced_distances(
        anchors,
        similar,
        inputs,
        metric,
        batch_size=distance_batch_size,
        pairwise_distances=pairwise_distances,
    )
    d_out = sliced_distances(
        anchors,
        outlier,
        inputs,
        metric,
        batch_size=distance_batch_size,
        pairwise_distances=pairwise_distances,
    )
    p_sim = -d_sim.square() / (sig[anchors] * sig[similar])
    p_out = -d_out.square() / (sig[anchors] * sig[outlier])
    flip = p_sim < p_out
    first = torch.where(flip, outlier, similar)
    second = torch.where(flip, similar, outlier)
    triplets = torch.stack((anchors, first, second), dim=1)
    return triplets, (p_sim - p_out).abs()


def find_scaled_neighbors(
    inputs: torch.Tensor,
    neighbors: torch.Tensor,
    metric: str,
    *,
    neighbor_distances: torch.Tensor | None = None,
    distance_batch_size: int | None = 1_000_000,
    pairwise_distances: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort neighbors by locally scaled squared distance and return the scales."""
    n, n_neighbors = neighbors.shape
    if n_neighbors < 2:
        raise ValueError("at least self and one neighbor are required")
    if neighbor_distances is None:
        anchors = torch.arange(n, device=neighbors.device)[:, None].expand_as(neighbors)
        neighbor_distances = sliced_distances(
            anchors.reshape(-1),
            neighbors.reshape(-1),
            inputs,
            metric,
            batch_size=distance_batch_size,
            pairwise_distances=pairwise_distances,
        ).reshape_as(neighbors)
    else:
        neighbor_distances = neighbor_distances.to(
            device=inputs.device, dtype=inputs.dtype
        )
        if neighbor_distances.shape != neighbors.shape:
            raise ValueError(
                "known neighbor indices and distances must have equal shapes"
            )

    scale_start = 3 if n_neighbors > 3 else 1
    scale_end = min(6, n_neighbors)
    sig = neighbor_distances[:, scale_start:scale_end].mean(dim=1).clamp_min(1e-10)
    scaled = neighbor_distances.square() / (sig[:, None] * sig[neighbors])

    # Self is canonicalized to column zero before this function.  Sort only the
    # remaining points so duplicate observations cannot displace it.
    order = torch.argsort(scaled[:, 1:], dim=1) + 1
    order = torch.cat(
        (torch.zeros((n, 1), dtype=torch.long, device=neighbors.device), order), dim=1
    )
    return torch.gather(scaled, 1, order), torch.gather(neighbors, 1, order), sig


def find_triplet_weights(
    inputs: torch.Tensor,
    triplets: torch.Tensor,
    inlier_neighbors: torch.Tensor,
    scaled_inlier_distances: torch.Tensor,
    sig: torch.Tensor,
    metric: str,
    *,
    distance_batch_size: int | None = 1_000_000,
    pairwise_distances: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate untransformed weights for nearest-neighbor triplets."""
    n, n_inliers = inlier_neighbors.shape
    expected_per_inlier = triplets.shape[0] // (n * n_inliers)
    if expected_per_inlier * n * n_inliers != triplets.shape[0]:
        raise ValueError("triplet count is incompatible with the neighbor layout")
    p_sim = (
        (-scaled_inlier_distances)[:, :, None]
        .expand(n, n_inliers, expected_per_inlier)
        .reshape(-1)
    )
    anchors, outliers = triplets[:, 0], triplets[:, 2]
    outlier_distances = sliced_distances(
        anchors,
        outliers,
        inputs,
        metric,
        batch_size=distance_batch_size,
        pairwise_distances=pairwise_distances,
    )
    p_out = -outlier_distances.square() / (sig[anchors] * sig[outliers])
    return p_sim - p_out


def _canonicalize_known_neighbors(
    neighbors: torch.Tensor, distances: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    n, width = neighbors.shape
    rows = torch.arange(n, device=neighbors.device)[:, None]
    self_mask = neighbors == rows
    self_count = self_mask.sum(dim=1)
    has_self = self_count > 0
    if not bool(has_self.all()) and bool(has_self.any()):
        raise ValueError(
            "known k-NN rows must either all include self or all omit self"
        )
    if bool(has_self.all()):
        if not bool((self_count == 1).all()):
            raise ValueError("known k-NN rows must contain self exactly once")
        if bool(self_mask[:, 0].all()):
            return neighbors.long(), distances
        nonself_indices = neighbors[~self_mask].reshape(n, width - 1)
        nonself_distances = distances[~self_mask].reshape(n, width - 1)
        self_distances = distances[self_mask].reshape(n, 1)
        return (
            torch.cat((rows, nonself_indices), dim=1).long(),
            torch.cat((self_distances, nonself_distances), dim=1),
        )

    target_width = min(n, width + 1)
    return (
        torch.cat((rows, neighbors[:, : target_width - 1]), dim=1).long(),
        torch.cat(
            (
                torch.zeros((n, 1), dtype=distances.dtype, device=distances.device),
                distances[:, : target_width - 1],
            ),
            dim=1,
        ),
    )


@torch.no_grad()
def generate_triplets(
    inputs: torch.Tensor,
    n_inliers: int,
    n_outliers: int,
    n_random: int,
    *,
    weight_temp: float = 0.5,
    metric: str = "euclidean",
    knn_backend: str = "auto",
    knn_tuple: tuple[torch.Tensor, torch.Tensor] | None = None,
    pairwise_distances: torch.Tensor | None = None,
    query_batch_size: int = 4096,
    database_batch_size: int = 16384,
    distance_batch_size: int = 1_000_000,
    ann_threshold: int = 50_000,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate and weight all nearest-neighbor and random TriMap triplets."""
    n = inputs.shape[0]
    if n_inliers >= n - 1:
        raise ValueError("n_inliers must be less than number of points minus one")
    if n < n_inliers + n_outliers + 1:
        raise ValueError("not enough points for the requested inliers and outliers")
    metric = canonical_metric(metric)

    if pairwise_distances is not None:
        if pairwise_distances.shape != (n, n):
            raise ValueError(
                "pairwise distance matrix must have shape (n_points, n_points)"
            )
        n_extra = min(n_inliers + 51, n)
        known_distances, known_neighbors = torch.topk(
            pairwise_distances, n_extra, dim=1, largest=False, sorted=True
        )
        known_neighbors, known_distances = _canonicalize_known_neighbors(
            known_neighbors, known_distances
        )
    elif knn_tuple is not None:
        known_neighbors, known_distances = knn_tuple
        known_neighbors = known_neighbors.to(inputs.device).long()
        known_distances = known_distances.to(device=inputs.device, dtype=inputs.dtype)
        known_neighbors, known_distances = _canonicalize_known_neighbors(
            known_neighbors, known_distances
        )
    else:
        n_extra = min(n_inliers + 51, n)
        known_neighbors, known_distances = knn_search(
            inputs,
            n_extra,
            metric=metric,
            backend=knn_backend,
            query_batch_size=query_batch_size,
            database_batch_size=database_batch_size,
            ann_threshold=ann_threshold,
        )

    if known_neighbors.shape[1] < n_inliers + 1:
        raise ValueError("known k-NN data does not contain enough neighbors")
    scaled_distances, sorted_neighbors, sig = find_scaled_neighbors(
        inputs,
        known_neighbors,
        metric,
        neighbor_distances=known_distances,
        distance_batch_size=distance_batch_size,
        pairwise_distances=pairwise_distances,
    )
    sorted_neighbors = sorted_neighbors[:, : n_inliers + 1]
    scaled_distances = scaled_distances[:, : n_inliers + 1]

    triplets = sample_knn_triplets(
        sorted_neighbors, n_inliers, n_outliers, generator=generator
    )
    weights = find_triplet_weights(
        inputs,
        triplets,
        sorted_neighbors[:, 1:],
        scaled_distances[:, 1:],
        sig,
        metric,
        distance_batch_size=distance_batch_size,
        pairwise_distances=pairwise_distances,
    )
    if n_random:
        random_triplets, random_weights = sample_random_triplets(
            inputs,
            n_random,
            sig,
            metric,
            generator=generator,
            distance_batch_size=distance_batch_size,
            pairwise_distances=pairwise_distances,
        )
        triplets = torch.cat((triplets, random_triplets))
        # Mirror both legacy branches.  The Annoy path scales random weights by
        # 0.1, while generate_triplets_known_knn historically leaves them
        # unscaled for supplied neighbors or a distance matrix.
        random_scale = (
            1.0
            if knn_tuple is not None or pairwise_distances is not None
            else _RAND_WEIGHT_SCALE
        )
        weights = torch.cat((weights, random_scale * random_weights))

    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights - weights.min()
    return triplets.long(), tempered_log(1.0 + weights, weight_temp)


def _triplet_terms(
    embedding: torch.Tensor, triplets: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    anchors = embedding[triplets[:, 0]]
    similar = embedding[triplets[:, 1]]
    outliers = embedding[triplets[:, 2]]
    y_ij = anchors - similar
    y_ik = anchors - outliers
    d_ij = 1.0 + y_ij.square().sum(dim=1)
    d_ik = 1.0 + y_ik.square().sum(dim=1)
    losses = weights * d_ij / (d_ij + d_ik)
    return losses, d_ij, d_ik, y_ij, y_ik


def trimap_loss(
    embedding: torch.Tensor,
    triplets: torch.Tensor,
    weights: torch.Tensor,
    *,
    reduction: str = "mean",
    batch_size: int | None = None,
) -> torch.Tensor:
    """Differentiable TriMap loss, optionally evaluated in bounded batches."""
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    if reduction == "none" and batch_size is not None:
        values = [
            _triplet_terms(embedding, triplets[part], weights[part])[0]
            for part in _slices(triplets.shape[0], batch_size)
        ]
        return torch.cat(values)
    if reduction == "none":
        return _triplet_terms(embedding, triplets, weights)[0]

    total = embedding.new_zeros(())
    for part in _slices(triplets.shape[0], batch_size):
        total = (
            total + _triplet_terms(embedding, triplets[part], weights[part])[0].sum()
        )
    return total if reduction == "sum" else total / triplets.shape[0]


@torch.no_grad()
def trimap_explicit_grad(
    embedding: torch.Tensor,
    triplets: torch.Tensor,
    weights: torch.Tensor,
    *,
    batch_size: int | None = 1_000_000,
    reduction: str = "sum",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the analytical gradient, loss, and number of violations.

    Unlike the historical Numba kernel, this includes the mathematically
    required factor of two and therefore agrees exactly with autograd.
    """
    if reduction not in {"sum", "mean"}:
        raise ValueError("explicit gradient reduction must be 'sum' or 'mean'")
    gradient = torch.zeros_like(embedding)
    total_loss = embedding.new_zeros(())
    violations = torch.zeros((), dtype=torch.long, device=embedding.device)
    for part in _slices(triplets.shape[0], batch_size):
        current = triplets[part]
        current_weights = weights[part]
        losses, d_ij, d_ik, y_ij, y_ik = _triplet_terms(
            embedding, current, current_weights
        )
        scale = (2.0 * current_weights / (d_ij + d_ik).square())[:, None]
        grad_similar = y_ij * d_ik[:, None] * scale
        grad_outlier = y_ik * d_ij[:, None] * scale
        gradient.index_add_(0, current[:, 0], grad_similar - grad_outlier)
        gradient.index_add_(0, current[:, 1], -grad_similar)
        gradient.index_add_(0, current[:, 2], grad_outlier)
        total_loss += losses.sum()
        violations += (d_ij > d_ik).sum()
    if reduction == "mean":
        gradient /= triplets.shape[0]
        total_loss /= triplets.shape[0]
    return gradient, total_loss, violations


def update_embedding_dbd(
    embedding: torch.Tensor,
    gradient: torch.Tensor,
    velocity: torch.Tensor,
    gain: torch.Tensor,
    learning_rate: float | torch.Tensor,
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One parallel delta-bar-delta optimizer update."""
    momentum = _FINAL_MOMENTUM if iteration > _SWITCH_ITER else _INIT_MOMENTUM
    gain = torch.where(
        torch.sign(velocity) != torch.sign(gradient),
        gain + _INCREASE_GAIN,
        torch.clamp(gain * _DAMP_GAIN, min=_MIN_GAIN),
    )
    velocity = momentum * velocity - learning_rate * gain * gradient
    return embedding + velocity, gain, velocity


def _pca_project(inputs: torch.Tensor, dimensions: int, seed: int) -> torch.Tensor:
    q = min(dimensions, inputs.shape[0], inputs.shape[1])
    if q < dimensions:
        raise ValueError(
            f"PCA initialization needs n_dims <= min(input shape), got {dimensions}"
        )
    devices = [inputs.device.index or 0] if inputs.is_cuda else []
    fork = (
        torch.random.fork_rng(devices=devices)
        if inputs.device.type != "mps"
        else contextlib.nullcontext()
    )
    with fork:
        torch.manual_seed(seed)
        if inputs.is_cuda:
            torch.cuda.manual_seed(seed)
        centered = inputs - inputs.mean(dim=0, keepdim=True)
        u, singular_values, _ = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    return u[:, :dimensions] * singular_values[:dimensions]


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "mps":
        torch.manual_seed(seed)
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


class TorchTRIMAP:
    """GPU-parallel TriMap estimator implemented with PyTorch.

    Parameters largely mirror :class:`trimap.TRIMAP`.  ``knn_backend='auto'``
    chooses cuVS CAGRA or Faiss when installed and appropriate, otherwise the
    exact blocked PyTorch implementation.  Set ``gradient='autograd'`` for the
    standard PyTorch derivative or ``gradient='explicit'`` for the lower-memory
    analytical kernel.
    """

    def __init__(
        self,
        n_dims: int = 2,
        n_inliers: int = 12,
        n_outliers: int = 4,
        n_random: int = 3,
        distance: str = "euclidean",
        lr: float = 0.1,
        n_iters: int = 400,
        triplets: Any | None = None,
        weights: Any | None = None,
        use_dist_matrix: bool = False,
        knn_tuple: tuple[Any, Any] | None = None,
        weight_temp: float = 0.5,
        apply_pca: bool = True,
        opt_method: str = "dbd",
        gradient: str = "explicit",
        init: str | Any = "pca",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        knn_backend: str = "auto",
        query_batch_size: int = 4096,
        database_batch_size: int = 16384,
        distance_batch_size: int = 1_000_000,
        triplet_batch_size: int = 1_000_000,
        ann_threshold: int = 50_000,
        random_state: int = 42,
        verbose: bool = False,
        return_seq: bool = False,
    ) -> None:
        self.n_dims = n_dims
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.distance = distance
        self.lr = lr
        self.n_iters = n_iters
        self.triplets = triplets
        self.weights = weights
        self.use_dist_matrix = use_dist_matrix
        self.knn_tuple = knn_tuple
        self.weight_temp = weight_temp
        self.apply_pca = apply_pca
        self.opt_method = opt_method
        self.gradient = gradient
        self.init = init
        self.device = device
        self.dtype = dtype
        self.knn_backend = knn_backend
        self.query_batch_size = query_batch_size
        self.database_batch_size = database_batch_size
        self.distance_batch_size = distance_batch_size
        self.triplet_batch_size = triplet_batch_size
        self.ann_threshold = ann_threshold
        self.random_state = random_state
        self.verbose = verbose
        self.return_seq = return_seq
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.n_dims < 2:
            raise ValueError("n_dims must be at least two")
        if self.n_inliers < 1 or self.n_outliers < 1 or self.n_random < 0:
            raise ValueError(
                "inliers/outliers must be positive and n_random non-negative"
            )
        if self.lr <= 0 or self.n_iters < 0:
            raise ValueError("lr must be positive and n_iters non-negative")
        if self.opt_method not in {"sd", "momentum", "dbd"}:
            raise ValueError("opt_method must be 'sd', 'momentum', or 'dbd'")
        if self.gradient not in {"autograd", "explicit"}:
            raise ValueError("gradient must be 'autograd' or 'explicit'")
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("dtype must be torch.float32 or torch.float64")
        canonical_metric(self.distance)

    def _tensor(
        self, value: Any, device: torch.device, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(value) if not isinstance(value, torch.Tensor) else value,
            device=device,
            dtype=dtype,
        )

    def _prepare_inputs(self, value: Any) -> tuple[torch.Tensor, torch.device]:
        if self.device is not None:
            device = torch.device(self.device)
        elif isinstance(value, torch.Tensor):
            device = value.device
        else:
            device = _default_device()
        inputs = self._tensor(value, device, dtype=self.dtype)
        if inputs.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("X contains NaN or infinite values")
        return inputs.contiguous(), device

    def _preprocess(self, inputs: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if self.use_dist_matrix or canonical_metric(self.distance) == "hamming":
            return inputs, False
        if inputs.shape[1] > _PCA_DIM and self.apply_pca:
            dimensions = min(_PCA_DIM, inputs.shape[0], inputs.shape[1])
            return _pca_project(inputs, dimensions, self.random_state), True
        minimum = inputs.amin()
        scale = (inputs.amax() - minimum).clamp_min(torch.finfo(inputs.dtype).eps)
        normalized = (inputs - minimum) / scale
        return normalized - normalized.mean(dim=0, keepdim=True), False

    def _initial_embedding(
        self,
        processed: torch.Tensor,
        init: str | Any,
        pca_solution: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        n = processed.shape[0]
        if isinstance(init, str):
            if init == "random":
                return _INIT_RANDOM_SCALE * torch.randn(
                    (n, self.n_dims),
                    device=processed.device,
                    dtype=processed.dtype,
                    generator=generator,
                )
            if init != "pca":
                raise ValueError("init must be 'pca', 'random', or an array")
            if pca_solution and processed.shape[1] >= self.n_dims:
                return _INIT_PCA_SCALE * processed[:, : self.n_dims].clone()
            return _INIT_PCA_SCALE * _pca_project(
                processed, self.n_dims, self.random_state
            )
        initial = self._tensor(init, processed.device, dtype=processed.dtype)
        if initial.shape != (n, self.n_dims):
            expected_shape = (n, self.n_dims)
            raise ValueError(
                f"initial embedding must have shape {expected_shape}, "
                f"got {initial.shape}"
            )
        return initial.clone()

    def _autograd_gradient(
        self,
        evaluation: torch.Tensor,
        triplets: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evaluation = evaluation.detach().requires_grad_(True)
        total_loss = evaluation.new_zeros(())
        violations = torch.zeros((), dtype=torch.long, device=evaluation.device)
        for part in _slices(triplets.shape[0], self.triplet_batch_size):
            losses, d_ij, d_ik, _, _ = _triplet_terms(
                evaluation, triplets[part], weights[part]
            )
            batch_loss = losses.sum()
            batch_loss.backward()
            total_loss += batch_loss.detach()
            violations += (d_ij.detach() > d_ik.detach()).sum()
        assert evaluation.grad is not None
        return evaluation.grad.detach(), total_loss, violations

    def _optimize(
        self,
        initial: torch.Tensor,
        triplets: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedding = initial.detach().clone()
        velocity = torch.zeros_like(embedding)
        gain = torch.ones_like(embedding)
        # The historical Numba gradient omits the derivative's factor of two.
        # Preserve the public learning-rate semantics of legacy TRIMAP while
        # retaining the mathematically correct PyTorch/autograd gradient.
        learning_rate = embedding.new_tensor(
            self.lr * _LEGACY_LEARNING_RATE_SCALE
        )
        old_loss = embedding.new_tensor(float("inf"))
        sequence = [embedding.clone()] if self.return_seq else None
        history: list[torch.Tensor] = []

        start_time = time.perf_counter()
        for iteration in range(self.n_iters):
            momentum = _FINAL_MOMENTUM if iteration > _SWITCH_ITER else _INIT_MOMENTUM
            evaluation = (
                embedding
                if self.opt_method == "sd"
                else embedding + momentum * velocity
            )
            if self.gradient == "autograd":
                gradient, loss_sum, violations = self._autograd_gradient(
                    evaluation, triplets, weights
                )
            else:
                gradient, loss_sum, violations = trimap_explicit_grad(
                    evaluation,
                    triplets,
                    weights,
                    batch_size=self.triplet_batch_size,
                )

            mean_loss = loss_sum / triplets.shape[0]
            history.append(mean_loss.detach())
            with torch.no_grad():
                if self.opt_method == "sd":
                    embedding = embedding - learning_rate * gradient
                elif self.opt_method == "momentum":
                    velocity = momentum * velocity - learning_rate * gradient
                    embedding = embedding + velocity
                else:
                    embedding, gain, velocity = update_embedding_dbd(
                        embedding,
                        gradient,
                        velocity,
                        gain,
                        learning_rate,
                        iteration,
                    )
                if self.opt_method != "dbd":
                    learning_rate = torch.where(
                        old_loss > mean_loss + 1e-7,
                        learning_rate * 1.01,
                        learning_rate * 0.9,
                    )
                old_loss = mean_loss

            if sequence is not None and (iteration + 1) % _RETURN_EVERY == 0:
                sequence.append(embedding.clone())
            if self.verbose and (iteration + 1) % _DISPLAY_EVERY == 0:
                print(
                    f"Iteration {iteration + 1:4d}/{self.n_iters:4d}, "
                    f"loss {mean_loss.item():.5f}, violated "
                    f"{100.0 * violations.item() / triplets.shape[0]:.2f}%"
                )

        if self.verbose:
            print(f"optimization finished in {time.perf_counter() - start_time:.2f}s")
        self.loss_history_ = (
            torch.stack(history) if history else torch.empty(0, device=embedding.device)
        )
        return embedding, torch.stack(sequence, dim=2) if sequence is not None else None

    def fit(self, X: Any, init: str | Any | None = None) -> TorchTRIMAP:
        """Fit a TriMap embedding while retaining tensors on the selected device."""
        inputs, device = self._prepare_inputs(X)
        n = inputs.shape[0]
        if self.n_inliers >= n - 1:
            raise ValueError("n_inliers must be less than number of points minus one")
        if n < self.n_inliers + self.n_outliers + 1:
            raise ValueError("not enough points for requested triplet sampling")
        if self.use_dist_matrix and inputs.shape[1] != n:
            raise ValueError("use_dist_matrix=True requires a square distance matrix")

        generator = _make_generator(device, self.random_state)
        # Match the legacy estimator: precomputed neighbors and stored
        # constraints are interpreted in the coordinate system supplied by the
        # caller.  Re-normalizing here would make neighbor distances and the
        # outlier distances used for weights live on different scales.
        if self.knn_tuple is not None or self.triplets is not None:
            processed, pca_solution = inputs, False
        else:
            processed, pca_solution = self._preprocess(inputs)
        self.device_ = device
        self.n_features_in_ = inputs.shape[1]
        self.preprocessed_ = processed

        if (self.triplets is None) != (self.weights is None):
            raise ValueError("triplets and weights must be supplied together")
        if self.triplets is None:
            known_knn = None
            if self.knn_tuple is not None:
                known_knn = (
                    self._tensor(self.knn_tuple[0], device, dtype=torch.long),
                    self._tensor(self.knn_tuple[1], device, dtype=self.dtype),
                )
            pairwise = inputs if self.use_dist_matrix else None
            self.knn_backend_ = (
                "precomputed"
                if pairwise is not None or known_knn is not None
                else resolve_knn_backend(
                    self.knn_backend,
                    processed,
                    self.distance,
                    ann_threshold=self.ann_threshold,
                )
            )
            triplets, weights = generate_triplets(
                processed,
                self.n_inliers,
                self.n_outliers,
                self.n_random,
                weight_temp=self.weight_temp,
                metric=self.distance,
                knn_backend=self.knn_backend,
                knn_tuple=known_knn,
                pairwise_distances=pairwise,
                query_batch_size=self.query_batch_size,
                database_batch_size=self.database_batch_size,
                distance_batch_size=self.distance_batch_size,
                ann_threshold=self.ann_threshold,
                generator=generator,
            )
        else:
            triplets = self._tensor(self.triplets, device, dtype=torch.long)
            weights = self._tensor(self.weights, device, dtype=self.dtype)
            if triplets.ndim != 2 or triplets.shape[1] != 3:
                raise ValueError("triplets must have shape (n_triplets, 3)")
            if weights.shape != (triplets.shape[0],):
                raise ValueError("weights must have one value per triplet")
            if bool((triplets < 0).any()) or bool((triplets >= n).any()):
                raise ValueError("triplet indices are out of range")
            self.knn_backend_ = "stored-triplets"

        self.triplets_ = self.triplets = triplets
        self.weights_ = self.weights = weights
        initial = self._initial_embedding(
            processed,
            self.init if init is None else init,
            pca_solution,
            generator,
        )
        final_embedding, sequence = self._optimize(initial, triplets, weights)
        self.final_embedding_ = final_embedding
        self.embedding_sequence_ = sequence
        self.embedding_ = sequence if self.return_seq else final_embedding
        return self

    def fit_transform(self, X: Any, init: str | Any | None = None) -> torch.Tensor:
        """Fit and return the device-resident embedding tensor."""
        return self.fit(X, init=init).embedding_

    def sample_triplets(self, X: Any) -> TorchTRIMAP:
        """Precompute and retain triplets without optimizing an embedding."""
        inputs, device = self._prepare_inputs(X)
        processed = (
            inputs if self.knn_tuple is not None else self._preprocess(inputs)[0]
        )
        generator = _make_generator(device, self.random_state)
        known_knn = None
        if self.knn_tuple is not None:
            known_knn = (
                self._tensor(self.knn_tuple[0], device, dtype=torch.long),
                self._tensor(self.knn_tuple[1], device, dtype=self.dtype),
            )
        self.triplets_, self.weights_ = generate_triplets(
            processed,
            self.n_inliers,
            self.n_outliers,
            self.n_random,
            weight_temp=self.weight_temp,
            metric=self.distance,
            knn_backend=self.knn_backend,
            knn_tuple=known_knn,
            pairwise_distances=inputs if self.use_dist_matrix else None,
            query_batch_size=self.query_batch_size,
            database_batch_size=self.database_batch_size,
            distance_batch_size=self.distance_batch_size,
            ann_threshold=self.ann_threshold,
            generator=generator,
        )
        self.triplets, self.weights = self.triplets_, self.weights_
        return self

    def del_triplets(self) -> TorchTRIMAP:
        """Delete retained triplets so the next fit resamples them."""
        self.triplets = self.weights = None
        for name in ("triplets_", "weights_"):
            if hasattr(self, name):
                delattr(self, name)
        return self

    @torch.no_grad()
    def global_score(self, X: Any, Y: Any) -> torch.Tensor:
        """Compute TriMap's global score on the estimator device."""
        inputs, device = self._prepare_inputs(X)
        embedding = self._tensor(Y, device, dtype=self.dtype)

        def global_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            source = source - source.mean(dim=0, keepdim=True)
            target = target - target.mean(dim=0, keepdim=True)
            transform = source.T @ target @ torch.linalg.pinv(target.T @ target)
            return (source.T - transform @ target.T).square().mean()

        pca = _pca_project(inputs, embedding.shape[1], self.random_state)
        pca_loss = global_loss(inputs, pca)
        embedding_loss = global_loss(inputs, embedding)
        return torch.exp(-(embedding_loss - pca_loss) / pca_loss)


def transform(X: Any, **kwargs: Any) -> torch.Tensor:
    """Functional shorthand for ``TorchTRIMAP(**kwargs).fit_transform(X)``."""
    return TorchTRIMAP(**kwargs).fit_transform(X)


__all__ = [
    "TorchTRIMAP",
    "find_scaled_neighbors",
    "find_triplet_weights",
    "generate_triplets",
    "rejection_sample",
    "sample_knn_triplets",
    "sample_random_triplets",
    "sliced_distances",
    "tempered_log",
    "transform",
    "trimap_explicit_grad",
    "trimap_loss",
    "update_embedding_dbd",
]
