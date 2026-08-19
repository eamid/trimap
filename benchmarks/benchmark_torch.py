"""Small reproducible throughput benchmark for the PyTorch implementation.

Example:
    python benchmarks/benchmark_torch.py --device cuda --n 100000 --triplets 5000000
"""

from __future__ import annotations

import argparse
import time

import torch

from trimap.torch_neighbors import torch_knn
from trimap.torch_trimap import trimap_explicit_grad, trimap_loss


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def elapsed(device: torch.device, function, repeats: int = 3) -> float:
    function()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    synchronize(device)
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=63)
    parser.add_argument("--triplets", type=int, default=1_000_000)
    parser.add_argument("--query-batch", type=int, default=2048)
    parser.add_argument("--database-batch", type=int, default=8192)
    parser.add_argument("--triplet-batch", type=int, default=500_000)
    parser.add_argument("--skip-knn", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(42)
    inputs = torch.randn(args.n, args.dim, device=device, generator=generator)
    embedding = torch.randn(args.n, 2, device=device, generator=generator) * 1e-4
    triplets = torch.randint(
        args.n, (args.triplets, 3), device=device, generator=generator
    )
    weights = torch.rand(args.triplets, device=device, generator=generator)

    if not args.skip_knn:
        knn_seconds = elapsed(
            device,
            lambda: torch_knn(
                inputs,
                min(args.k, args.n),
                query_batch_size=args.query_batch,
                database_batch_size=args.database_batch,
            ),
            repeats=1,
        )
        print(
            f"exact blocked k-NN: {knn_seconds:.3f}s ({args.n / knn_seconds:,.0f} queries/s)"
        )

    def explicit() -> None:
        trimap_explicit_grad(
            embedding, triplets, weights, batch_size=args.triplet_batch
        )

    def autograd() -> None:
        current = embedding.detach().requires_grad_(True)
        for start in range(0, args.triplets, args.triplet_batch):
            trimap_loss(
                current,
                triplets[start : start + args.triplet_batch],
                weights[start : start + args.triplet_batch],
                reduction="sum",
            ).backward()

    explicit_seconds = elapsed(device, explicit)
    autograd_seconds = elapsed(device, autograd)
    print(
        f"explicit gradient: {explicit_seconds:.3f}s "
        f"({args.triplets / explicit_seconds:,.0f} triplets/s)"
    )
    print(
        f"autograd gradient: {autograd_seconds:.3f}s "
        f"({args.triplets / autograd_seconds:,.0f} triplets/s)"
    )


if __name__ == "__main__":
    main()
