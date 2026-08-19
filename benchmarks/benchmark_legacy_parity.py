"""Benchmark accelerated optimization against legacy TriMap on identical state.

Example:
    python benchmarks/benchmark_legacy_parity.py --device mps --n 10000
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from trimap import TRIMAP, TorchTRIMAP


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def make_constraints(
    n: int, n_inliers: int, n_outliers: int, n_random: int
) -> np.ndarray:
    anchors = np.arange(n, dtype=np.int32)[:, None, None]
    inlier_offsets = np.arange(1, n_inliers + 1, dtype=np.int32)[None, :, None]
    outlier_offsets = (
        n_inliers
        + 1
        + np.arange(n_inliers * n_outliers, dtype=np.int32).reshape(
            1, n_inliers, n_outliers
        )
    )
    knn = np.stack(
        (
            np.broadcast_to(anchors, (n, n_inliers, n_outliers)),
            np.broadcast_to((anchors + inlier_offsets) % n, (n, n_inliers, n_outliers)),
            (anchors + outlier_offsets) % n,
        ),
        axis=-1,
    ).reshape(-1, 3)
    if not n_random:
        return knn
    random_offsets = np.arange(n_random, dtype=np.int32)[None, :]
    random = np.stack(
        (
            np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], (n, n_random)),
            (np.arange(n, dtype=np.int32)[:, None] + n_inliers + 7 + random_offsets)
            % n,
            (
                np.arange(n, dtype=np.int32)[:, None]
                + 2 * n_inliers
                + 19
                + random_offsets
            )
            % n,
        ),
        axis=-1,
    ).reshape(-1, 3)
    return np.concatenate((knn, random))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--n-inliers", type=int, default=12)
    parser.add_argument("--n-outliers", type=int, default=4)
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=400)
    parser.add_argument("--triplet-batch", type=int, default=250_000)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    minimum_n = 2 * args.n_inliers + args.n_outliers + args.n_random + 20
    if args.n <= minimum_n:
        raise ValueError(f"n must be greater than {minimum_n}")

    rng = np.random.default_rng(42)
    inputs = rng.normal(size=(args.n, args.dim)).astype(np.float32)
    initial = (rng.normal(size=(args.n, 2)) * 1e-4).astype(np.float32)
    triplets = make_constraints(
        args.n, args.n_inliers, args.n_outliers, args.n_random
    )
    weights = rng.uniform(0.05, 3.0, len(triplets)).astype(np.float32)
    common = dict(
        n_dims=2,
        n_inliers=args.n_inliers,
        n_outliers=args.n_outliers,
        n_random=args.n_random,
        lr=0.1,
        n_iters=args.n_iters,
        triplets=triplets,
        weights=weights,
        opt_method="dbd",
    )

    # Compile Numba and warm accelerator kernels outside the timed region.
    warm_common = {**common, "n_iters": 1}
    TRIMAP(**warm_common).fit_transform(inputs.copy(), init=initial.copy())
    TorchTRIMAP(
        **warm_common,
        device=device,
        gradient="explicit",
        triplet_batch_size=args.triplet_batch,
    ).fit_transform(inputs, init=initial)
    synchronize(device)

    start = time.perf_counter()
    legacy = TRIMAP(**common).fit_transform(inputs.copy(), init=initial.copy())
    legacy_seconds = time.perf_counter() - start

    start = time.perf_counter()
    accelerated = TorchTRIMAP(
        **common,
        device=device,
        gradient="explicit",
        triplet_batch_size=args.triplet_batch,
    ).fit_transform(inputs, init=initial)
    synchronize(device)
    accelerated_seconds = time.perf_counter() - start
    accelerated_numpy = accelerated.cpu().numpy()
    difference = accelerated_numpy.astype(np.float64) - legacy.astype(np.float64)

    print(
        json.dumps(
            {
                "device": str(device),
                "n": args.n,
                "triplets": len(triplets),
                "iterations": args.n_iters,
                "legacy_seconds": legacy_seconds,
                "accelerated_seconds": accelerated_seconds,
                "speedup": legacy_seconds / accelerated_seconds,
                "parity_rmse": float(np.sqrt(np.mean(difference**2))),
                "parity_max_abs": float(np.max(np.abs(difference))),
                "parity_relative_rmse": float(
                    np.sqrt(np.mean(difference**2))
                    / np.sqrt(np.mean(legacy.astype(np.float64) ** 2))
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
