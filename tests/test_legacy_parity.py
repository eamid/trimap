import numpy as np
import pytest
import torch

from trimap import TRIMAP, TorchTRIMAP
from trimap.torch_trimap import find_scaled_neighbors, find_triplet_weights
from trimap.trimap_ import find_p, find_weights


@pytest.mark.parametrize("opt_method", ["sd", "momentum", "dbd"])
def test_stored_constraint_optimization_matches_legacy(opt_method: str) -> None:
    rng = np.random.default_rng(123)
    n = 32
    n_inliers = 2
    n_outliers = 2
    triplets = np.asarray(
        [
            (anchor, (anchor + inlier + 1) % n, (anchor + 8 + 3 * inlier + outlier) % n)
            for anchor in range(n)
            for inlier in range(n_inliers)
            for outlier in range(n_outliers)
        ],
        dtype=np.int32,
    )
    weights = rng.uniform(0.1, 2.0, len(triplets)).astype(np.float32)
    inputs = rng.normal(size=(n, 8)).astype(np.float32)
    initial = (rng.normal(size=(n, 2)) * 1e-4).astype(np.float32)
    common = dict(
        n_dims=2,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=0,
        lr=0.1,
        n_iters=11,
        triplets=triplets,
        weights=weights,
        opt_method=opt_method,
        return_seq=True,
    )

    legacy = TRIMAP(**common).fit_transform(inputs.copy(), init=initial.copy())
    accelerated = (
        TorchTRIMAP(
            **common,
            device="cpu",
            gradient="explicit",
            triplet_batch_size=37,
        )
        .fit_transform(inputs, init=initial)
        .numpy()
    )

    np.testing.assert_allclose(accelerated, legacy, atol=5e-7, rtol=5e-7)


def test_weight_formula_matches_legacy_for_identical_triplets() -> None:
    rng = np.random.default_rng(321)
    inputs = rng.normal(size=(24, 6)).astype(np.float32)
    pairwise = np.linalg.norm(inputs[:, None] - inputs[None, :], axis=2)
    neighbors = np.argsort(pairwise, axis=1)[:, :8].astype(np.int32)
    neighbor_distances = np.take_along_axis(pairwise, neighbors, axis=1).astype(
        np.float32
    )
    sig = np.maximum(neighbor_distances[:, 3:6].mean(axis=1), 1e-10)
    similarities = find_p(neighbor_distances, sig, neighbors)
    legacy_order = np.argsort(-similarities, axis=1)
    legacy_neighbors = np.take_along_axis(neighbors, legacy_order, axis=1)[:, :4]

    torch_scaled, torch_neighbors, torch_sig = find_scaled_neighbors(
        torch.from_numpy(inputs),
        torch.from_numpy(neighbors),
        "euclidean",
        neighbor_distances=torch.from_numpy(neighbor_distances),
    )
    np.testing.assert_array_equal(torch_neighbors[:, :4].numpy(), legacy_neighbors)

    triplets = np.asarray(
        [
            (anchor, legacy_neighbors[anchor, inlier], (anchor + 11 + outlier) % 24)
            for anchor in range(24)
            for inlier in range(1, 4)
            for outlier in range(2)
        ],
        dtype=np.int32,
    )
    outlier_distances = np.linalg.norm(
        inputs[triplets[:, 0]] - inputs[triplets[:, 2]], axis=1
    ).astype(np.float32)
    legacy_weights = find_weights(
        triplets, similarities, neighbors, outlier_distances, sig
    )
    accelerated_weights = find_triplet_weights(
        torch.from_numpy(inputs),
        torch.from_numpy(triplets).long(),
        torch_neighbors[:, 1:4],
        torch_scaled[:, 1:4],
        torch_sig,
        "euclidean",
    ).numpy()

    np.testing.assert_allclose(
        accelerated_weights, legacy_weights, atol=2e-5, rtol=2e-5
    )


def test_dbd_parity_across_momentum_switch() -> None:
    rng = np.random.default_rng(456)
    n = 16
    triplets = np.asarray(
        [(anchor, (anchor + 1) % n, (anchor + 7) % n) for anchor in range(n)],
        dtype=np.int32,
    )
    weights = rng.uniform(0.1, 1.0, n).astype(np.float32)
    inputs = rng.normal(size=(n, 4)).astype(np.float32)
    initial = (rng.normal(size=(n, 2)) * 1e-4).astype(np.float32)
    common = dict(
        n_inliers=1,
        n_outliers=1,
        n_random=0,
        n_iters=252,
        lr=0.1,
        triplets=triplets,
        weights=weights,
        opt_method="dbd",
    )

    legacy = TRIMAP(**common).fit_transform(inputs.copy(), init=initial.copy())
    accelerated = (
        TorchTRIMAP(**common, device="cpu", gradient="explicit")
        .fit_transform(inputs, init=initial)
        .numpy()
    )

    np.testing.assert_allclose(accelerated, legacy, atol=1e-6, rtol=1e-6)
