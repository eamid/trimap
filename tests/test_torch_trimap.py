import numpy as np
import pytest
import torch

from trimap import TorchTRIMAP
from trimap.torch_neighbors import torch_knn
from trimap.torch_trimap import (
    generate_triplets,
    rejection_sample,
    tempered_log,
    trimap_explicit_grad,
    trimap_loss,
)


def test_tempered_log_limits() -> None:
    values = torch.tensor([1.0, 2.0, 4.0])
    torch.testing.assert_close(tempered_log(values, 0.0), values - 1.0)
    torch.testing.assert_close(tempered_log(values, 1.0), values.log())


def test_rejection_sample_is_unique_and_respects_rows() -> None:
    rejects = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    samples = rejection_sample(
        rejects, 4, 20, generator=torch.Generator().manual_seed(2)
    )
    assert samples.shape == (3, 4)
    assert not bool((samples[:, :, None] == rejects[:, None, :]).any())
    assert all(torch.unique(row).numel() == row.numel() for row in samples)


def test_generate_triplets_shapes_and_ordering() -> None:
    inputs = torch.randn(36, 7, generator=torch.Generator().manual_seed(3))
    triplets, weights = generate_triplets(
        inputs,
        4,
        3,
        2,
        knn_backend="torch",
        query_batch_size=11,
        database_batch_size=13,
        generator=torch.Generator().manual_seed(5),
    )
    assert triplets.shape == (36 * (4 * 3 + 2), 3)
    assert weights.shape == (triplets.shape[0],)
    assert triplets.dtype == torch.long
    assert bool(torch.isfinite(weights).all())
    assert bool((weights >= 0).all())
    assert not bool((triplets[:, 0] == triplets[:, 1]).any())
    assert not bool((triplets[:, 0] == triplets[:, 2]).any())
    assert not bool((triplets[:, 1] == triplets[:, 2]).any())


def test_explicit_gradient_matches_autograd_and_batching() -> None:
    embedding = torch.randn(
        9, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(7)
    ).requires_grad_(True)
    triplets = torch.tensor(
        [[0, 1, 2], [0, 3, 4], [2, 5, 1], [7, 4, 8], [6, 3, 0], [8, 2, 4]]
    )
    weights = torch.tensor([0.2, 1.1, 0.7, 2.0, 0.1, 0.9], dtype=torch.float64)

    loss = trimap_loss(embedding, triplets, weights, reduction="sum")
    loss.backward()
    explicit, explicit_loss, violations = trimap_explicit_grad(
        embedding.detach(), triplets, weights, batch_size=2
    )
    torch.testing.assert_close(explicit_loss, loss.detach(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(explicit, embedding.grad, atol=1e-12, rtol=1e-12)
    assert 0 <= violations.item() <= triplets.shape[0]

    unbatched = trimap_loss(embedding.detach(), triplets, weights, reduction="mean")
    batched = trimap_loss(
        embedding.detach(), triplets, weights, reduction="mean", batch_size=2
    )
    torch.testing.assert_close(batched, unbatched, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("gradient", ["autograd", "explicit"])
def test_estimator_runs_full_pipeline(gradient: str) -> None:
    inputs = torch.randn(48, 10, generator=torch.Generator().manual_seed(12))
    estimator = TorchTRIMAP(
        n_inliers=4,
        n_outliers=2,
        n_random=1,
        n_iters=4,
        init="random",
        gradient=gradient,
        knn_backend="torch",
        query_batch_size=13,
        database_batch_size=17,
        triplet_batch_size=100,
        random_state=8,
    )
    embedding = estimator.fit_transform(inputs)
    assert embedding.shape == (48, 2)
    assert embedding.device == inputs.device
    assert bool(torch.isfinite(embedding).all())
    assert estimator.triplets_.shape == (48 * (4 * 2 + 1), 3)
    assert estimator.loss_history_.shape == (4,)
    assert estimator.knn_backend_ == "torch"


def test_autograd_and_explicit_optimization_agree() -> None:
    inputs = torch.randn(32, 6, generator=torch.Generator().manual_seed(13))
    triplets, weights = generate_triplets(
        inputs,
        3,
        2,
        1,
        knn_backend="torch",
        generator=torch.Generator().manual_seed(14),
    )
    initial = torch.randn(32, 2, generator=torch.Generator().manual_seed(15)) * 1e-4
    common = {
        "n_inliers": 3,
        "n_outliers": 2,
        "n_random": 1,
        "n_iters": 3,
        "triplets": triplets,
        "weights": weights,
        "triplet_batch_size": 37,
    }
    automatic = TorchTRIMAP(**common, gradient="autograd").fit_transform(
        inputs, init=initial
    )
    explicit = TorchTRIMAP(**common, gradient="explicit").fit_transform(
        inputs, init=initial
    )
    torch.testing.assert_close(automatic, explicit, atol=2e-6, rtol=2e-5)


def test_learning_rate_has_legacy_step_size() -> None:
    inputs = torch.randn(9, 4, generator=torch.Generator().manual_seed(19))
    initial = torch.randn(9, 2, generator=torch.Generator().manual_seed(20))
    triplets = torch.tensor(
        [[0, 1, 2], [0, 3, 4], [2, 5, 1], [7, 4, 8], [6, 3, 0], [8, 2, 4]]
    )
    weights = torch.tensor([0.2, 1.1, 0.7, 2.0, 0.1, 0.9])
    gradient, _, _ = trimap_explicit_grad(initial, triplets, weights)

    result = TorchTRIMAP(
        n_inliers=3,
        n_outliers=2,
        n_random=0,
        lr=0.1,
        n_iters=1,
        triplets=triplets,
        weights=weights,
        opt_method="sd",
        gradient="explicit",
    ).fit_transform(inputs, init=initial)

    torch.testing.assert_close(result, initial - 0.05 * gradient)


def test_known_knn_without_self_and_precomputed_distances() -> None:
    inputs = torch.randn(25, 5, generator=torch.Generator().manual_seed(16))
    neighbors, distances = torch_knn(inputs, 6, query_batch_size=8)
    known = TorchTRIMAP(
        n_inliers=3,
        n_outliers=2,
        n_random=0,
        n_iters=1,
        init="random",
        knn_tuple=(neighbors[:, 1:], distances[:, 1:]),
    ).fit_transform(inputs)
    assert known.shape == (25, 2)

    pairwise = torch.cdist(inputs, inputs)
    precomputed = TorchTRIMAP(
        n_inliers=3,
        n_outliers=2,
        n_random=0,
        n_iters=1,
        init="random",
        use_dist_matrix=True,
    ).fit_transform(pairwise)
    assert precomputed.shape == (25, 2)


def test_known_knn_preserves_the_supplied_feature_scale() -> None:
    inputs = 7.0 + 3.0 * torch.randn(
        25, 5, generator=torch.Generator().manual_seed(18)
    )
    neighbors, distances = torch_knn(inputs, 6, query_batch_size=8)
    estimator = TorchTRIMAP(
        n_inliers=3,
        n_outliers=2,
        n_random=0,
        n_iters=0,
        knn_tuple=(neighbors, distances),
    ).fit(inputs, init=torch.zeros(25, 2))

    torch.testing.assert_close(estimator.preprocessed_, inputs)


def test_numpy_input_and_sequence_output() -> None:
    inputs = np.random.default_rng(17).normal(size=(24, 5)).astype(np.float32)
    result = TorchTRIMAP(
        n_inliers=3,
        n_outliers=2,
        n_random=0,
        n_iters=11,
        init="random",
        return_seq=True,
        query_batch_size=10,
    ).fit_transform(inputs)
    assert result.shape == (24, 2, 2)
    assert isinstance(result, torch.Tensor)
