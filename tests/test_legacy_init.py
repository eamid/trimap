import numpy as np

from trimap import TRIMAP


def test_legacy_accepts_array_initialization() -> None:
    inputs = np.random.default_rng(1).normal(size=(8, 4)).astype(np.float32)
    initialization = np.random.default_rng(2).normal(size=(8, 2)).astype(np.float32)
    triplets = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    weights = np.ones(2, dtype=np.float32)
    result = TRIMAP(
        n_inliers=1,
        n_outliers=1,
        n_random=0,
        n_iters=0,
        triplets=triplets,
        weights=weights,
    ).fit_transform(inputs, init=initialization)
    np.testing.assert_array_equal(result, initialization)
