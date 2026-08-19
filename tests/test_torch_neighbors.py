import pytest
import torch

from trimap.torch_neighbors import (
    canonical_metric,
    resolve_knn_backend,
    rowwise_distances,
    torch_knn,
)


def test_rowwise_distances() -> None:
    x1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    x2 = torch.tensor([[3.0, 4.0], [0.0, 1.0], [1.0, 0.0]])
    torch.testing.assert_close(
        rowwise_distances(x1, x2, "euclidean"), torch.tensor([5.0, 2**0.5, 1.0])
    )
    torch.testing.assert_close(
        rowwise_distances(x1, x2, "manhattan"), torch.tensor([7.0, 2.0, 1.0])
    )
    torch.testing.assert_close(
        rowwise_distances(x1, x2, "hamming"), torch.tensor([2.0, 2.0, 1.0])
    )
    torch.testing.assert_close(
        rowwise_distances(x1, x2, "chebyshev"), torch.tensor([4.0, 1.0, 1.0])
    )
    assert canonical_metric("angular") == "cosine"


@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "cosine", "hamming", "chebyshev"]
)
def test_torch_knn_is_sorted_and_contains_self(metric: str) -> None:
    generator = torch.Generator().manual_seed(10)
    if metric == "hamming":
        inputs = torch.randint(0, 2, (19, 6), generator=generator).float()
    else:
        inputs = torch.randn(19, 6, generator=generator)
    indices, distances = torch_knn(
        inputs,
        5,
        metric=metric,
        query_batch_size=4,
        database_batch_size=7,
    )
    torch.testing.assert_close(indices[:, 0], torch.arange(19))
    torch.testing.assert_close(distances[:, 0], torch.zeros(19))
    assert bool((distances[:, 1:] >= distances[:, :-1] - 1e-6).all())


def test_blocked_knn_matches_full_cdist() -> None:
    inputs = torch.randn(23, 8, generator=torch.Generator().manual_seed(4))
    indices, distances = torch_knn(inputs, 6, query_batch_size=5, database_batch_size=9)
    expected_distances, expected_indices = torch.topk(
        torch.cdist(inputs, inputs), 6, dim=1, largest=False, sorted=True
    )
    torch.testing.assert_close(distances, expected_distances, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(indices, expected_indices)


def test_auto_backend_always_has_native_fallback() -> None:
    inputs = torch.randn(10, 3)
    assert resolve_knn_backend("auto", inputs, "manhattan") == "torch"
    assert resolve_knn_backend("torch", inputs, "euclidean") == "torch"
    with pytest.raises(ValueError, match="Unknown knn_backend"):
        resolve_knn_backend("annoy", inputs, "euclidean")
