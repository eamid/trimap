#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

TriMap: Large-scale Dimensionality Reduction Using Triplet Constraints
Source: https://arxiv.org/pdf/1910.00204.pdf

"""


from sklearn.base import BaseEstimator
from numba import njit, prange
from annoy import AnnoyIndex
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import numpy as np
import time
import datetime
import sys
import warnings

# default hyperparams
_RAND_WEIGHT_SCALE = 0.1

_INIT_PCA_SCALE = 0.01
_INIT_RAND_SCALE = 0.0001

_PCA_DIM = 100
_NUM_TREES = 20
_NUM_EXTRA_KNN = 50

_FINAL_MOMENTUM = 0.8
_INIT_MOMENTUM = 0.5
_SWITCH_ITER_MOMENTUM = 250

_MIN_GAIN = 0.01
_DAMP_GAIN = 0.8
_INCREASE_GAIN = 0.2

_RETURN_EVERY = 10
_DISP_ITER = 100

if sys.version_info < (3,):
    range = xrange

_BOLD = "\033[1m"
_RESET = "\033[0;0m"


def tempered_log(x, t):
    """Tempered log with temperature t"""
    if np.abs(t - 1.0) < 1e-5:
        return np.log(x)
    else:
        return 1.0 / (1.0 - t) * (np.power(x, 1.0 - t) - 1.0)


@njit("f4(f4[:])")
def l2_norm(x):
    """L2 norm of a vector."""
    result = 0.0
    for i in range(x.shape[0]):
        result += x[i] ** 2
    return np.sqrt(result)


@njit("f4(f4[:],f4[:])")
def euclid_dist(x1, x2):
    """Euclidean distance between two vectors."""
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i] - x2[i]) ** 2
    return np.sqrt(result)


@njit("f4(f4[:],f4[:])")
def manhattan_dist(x1, x2):
    """Manhattan distance between two vectors."""
    result = 0.0
    for i in range(x1.shape[0]):
        result += np.abs(x1[i] - x2[i])
    return result


@njit("f4(f4[:],f4[:])")
def angular_dist(x1, x2):
    """Angular (i.e. cosine) distance between two vectors."""
    x1_norm = np.maximum(l2_norm(x1), 1e-20)
    x2_norm = np.maximum(l2_norm(x2), 1e-20)
    result = 0.0
    for i in range(x1.shape[0]):
        result += x1[i] * x2[i]
    # angular is multiplied by a factor of 2.0 in annoy
    return 2.0 * (1.0 - result / x1_norm / x2_norm)


@njit("f4(f4[:],f4[:])")
def hamming_dist(x1, x2):
    """Hamming distance between two vectors."""
    result = 0.0
    for i in range(x1.shape[0]):
        if x1[i] != x2[i]:
            result += 1.0
    return result


@njit()
def calculate_dist(x1, x2, distance_index):
    if distance_index == 0:
        return euclid_dist(x1, x2)
    elif distance_index == 1:
        return manhattan_dist(x1, x2)
    elif distance_index == 2:
        return angular_dist(x1, x2)
    elif distance_index == 3:
        return hamming_dist(x1, x2)


@njit()
def rejection_sample(n_samples, max_int, rejects):
    """Rejection sampling.

    Samples "n_samples" integers from a given interval [0,max_int] while
    rejecting the values that are in the "rejects".

    """
    result = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(max_int)
            for k in range(i):
                if j == result[k]:
                    break
            for k in range(rejects.shape[0]):
                if j == rejects[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result


@njit("i4[:,:](f4[:,:],i4[:,:],i4,i4)", parallel=True, nogil=True)
def sample_knn_triplets(P, nbrs, n_inliers, n_outliers):
    """Sample nearest neighbors triplets based on the similarity values in P.

    Input
    ------

    nbrs: Nearest neighbors indices for each point. The similarity values
        are given in matrix P. Row i corresponds to the i-th point.

    P: Matrix of pairwise similarities between each point and its neighbors
        given in matrix nbrs

    n_inliers: Number of inlier points

    n_outliers: Number of outlier points

    Output
    ------

    triplets: Sampled triplets
    """
    n, n_neighbors = nbrs.shape
    triplets = np.empty((n * n_inliers * n_outliers, 3), dtype=np.int32)
    for i in prange(n):
        sort_indices = np.argsort(-P[i])
        for j in prange(n_inliers):
            sim = nbrs[i][sort_indices[j + 1]]
            samples = rejection_sample(n_outliers, n, nbrs[i][sort_indices[: j + 2]])
            for k in prange(n_outliers):
                index = i * n_inliers * n_outliers + j * n_outliers + k
                out = samples[k]
                triplets[index][0] = i
                triplets[index][1] = sim
                triplets[index][2] = out
    return triplets


@njit("f4[:,:](f4[:,:],i4,f4[:],i4)", parallel=True, nogil=True)
def sample_random_triplets(X, n_random, sig, distance_index):
    """Sample uniformly random triplets.

    Input
    ------

    X: Instance matrix or pairwise distances

    n_random: Number of random triplets per point

    sig: Scaling factor for the distances

    distance_index: index of the distance measure

    Output
    ------

    rand_triplets: Sampled triplets
    """
    n = X.shape[0]
    rand_triplets = np.empty((n * n_random, 4), dtype=np.float32)
    for i in prange(n):
        for j in prange(n_random):
            sim = np.random.choice(n)
            while sim == i:
                sim = np.random.choice(n)
            out = np.random.choice(n)
            while out == i or out == sim:
                out = np.random.choice(n)
            if distance_index == -1:
                d_sim = X[i, sim]
            else:
                d_sim = calculate_dist(X[i], X[sim], distance_index)
            p_sim = -(d_sim**2) / (sig[i] * sig[sim])
            if distance_index == -1:
                d_out = X[i, out]
            else:
                d_out = calculate_dist(X[i], X[out], distance_index)
            p_out = -(d_out**2) / (sig[i] * sig[out])
            if p_sim < p_out:
                sim, out = out, sim
                p_sim, p_out = p_out, p_sim
            rand_triplets[i * n_random + j][0] = i
            rand_triplets[i * n_random + j][1] = sim
            rand_triplets[i * n_random + j][2] = out
            rand_triplets[i * n_random + j][3] = p_sim - p_out
    return rand_triplets


@njit("f4[:,:](f4[:,:],f4[:],i4[:,:])", parallel=True, nogil=True)
def find_p(knn_distances, sig, nbrs):
    """Calculates the similarity matrix P.

    Input
    ------

    knn_distances: Matrix of pairwise knn distances

    sig: Scaling factor for the distances

    nbrs: Nearest neighbors

    Output
    ------

    P: Pairwise similarity matrix
    """
    n, n_neighbors = knn_distances.shape
    P = np.zeros((n, n_neighbors), dtype=np.float32)
    for i in prange(n):
        for j in prange(n_neighbors):
            P[i][j] = -knn_distances[i][j] ** 2 / (sig[i] * sig[nbrs[i][j]])
    return P


@njit("f4[:](i4[:,:],f4[:,:],i4[:,:],f4[:],f4[:])", parallel=True, nogil=True)
def find_weights(triplets, P, nbrs, outlier_distances, sig):
    """Calculates the weights for the sampled nearest neighbors triplets.

    Input
    ------

    triplets: Sampled triplets

    P: Pairwise similarity matrix

    nbrs: Nearest neighbors

    outlier_distances: Matrix of pairwise outlier distances

    sig: Scaling factor for the distances

    Output
    ------

    weights: Weights for the triplets
    """
    n_triplets = triplets.shape[0]
    weights = np.empty(n_triplets, dtype=np.float32)
    for t in prange(n_triplets):
        i = triplets[t][0]
        sim = 0
        while nbrs[i][sim] != triplets[t][1]:
            sim += 1
        p_sim = P[i][sim]
        p_out = -outlier_distances[t] ** 2 / (sig[i] * sig[triplets[t][2]])
        weights[t] = p_sim - p_out
    return weights


def generate_triplets(
    X,
    n_inliers,
    n_outliers,
    n_random,
    distance="euclidean",
    verbose=False,
    weight_temp=0.5,
):
    distance_dict = {"euclidean": 0, "manhattan": 1, "angular": 2, "hamming": 3}
    distance_index = distance_dict[distance]
    n, dim = X.shape
    n_extra = min(n_inliers + _NUM_EXTRA_KNN, n)
    tree = AnnoyIndex(dim, metric=distance)
    for i in range(n):
        tree.add_item(i, X[i, :])
    tree.build(_NUM_TREES)
    nbrs = np.empty((n, n_extra), dtype=np.int32)
    knn_distances = np.empty((n, n_extra), dtype=np.float32)
    for i in range(n):
        nbrs[i, :] = tree.get_nns_by_item(i, n_extra)
        for j in range(n_extra):
            knn_distances[i, j] = tree.get_distance(i, nbrs[i, j])
    if verbose:
        print("found nearest neighbors")
    sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)  # scale parameter
    P = find_p(knn_distances, sig, nbrs)
    triplets = sample_knn_triplets(P, nbrs, n_inliers, n_outliers)
    n_triplets = triplets.shape[0]
    outlier_distances = np.empty(n_triplets, dtype=np.float32)
    for t in range(n_triplets):
        outlier_distances[t] = calculate_dist(
            X[triplets[t, 0], :], X[triplets[t, 2], :], distance_index
        )
    weights = find_weights(triplets, P, nbrs, outlier_distances, sig)
    if n_random > 0:
        rand_triplets = sample_random_triplets(X, n_random, sig, distance_index)
        rand_weights = rand_triplets[:, -1]
        rand_triplets = rand_triplets[:, :-1].astype(np.int32)
        triplets = np.vstack((triplets, rand_triplets))
        weights = np.hstack((weights, _RAND_WEIGHT_SCALE * rand_weights))
    weights[np.isnan(weights)] = 0.0
    weights -= np.min(weights)
    weights = tempered_log(1.0 + weights, weight_temp)
    return (triplets, weights)


def generate_triplets_known_knn(
    X,
    knn_nbrs,
    knn_distances,
    n_inliers,
    n_outliers,
    n_random,
    pairwise_dist_matrix=None,
    distance="euclidean",
    verbose=False,
    weight_temp=0.5,
):
    all_distances = pairwise_dist_matrix is not None
    if all_distances:
        distance = "other"
    distance_dict = {
        "euclidean": 0,
        "manhattan": 1,
        "angular": 2,
        "hamming": 3,
        "other": -1,
    }
    distance_index = distance_dict[distance]
    # check whether the first nn of each point is itself
    # TODO(eamid): use index shifting instead
    if knn_nbrs[0, 0] != 0:
        knn_nbrs = np.hstack(
            (np.array(range(knn_nbrs.shape[0]))[:, np.newaxis], knn_nbrs)
        ).astype(np.int32)
        knn_distances = np.hstack(
            (np.zeros((knn_distances.shape[0], 1)), knn_distances)
        ).astype(np.float32)
    sig = np.maximum(np.mean(knn_distances[:, 3:6], axis=1), 1e-10)  # scale parameter
    P = find_p(knn_distances, sig, knn_nbrs)
    triplets = sample_knn_triplets(P, knn_nbrs, n_inliers, n_outliers)
    n_triplets = triplets.shape[0]
    outlier_distances = np.empty(n_triplets, dtype=np.float32)
    for t in range(n_triplets):
        if all_distances:
            outlier_distances[t] = pairwise_dist_matrix[triplets[t, 0], triplets[t, 2]]
        else:
            outlier_distances[t] = calculate_dist(
                X[triplets[t, 0], :], X[triplets[t, 2], :], distance_index
            )
    weights = find_weights(triplets, P, knn_nbrs, outlier_distances, sig)
    if n_random > 0:
        if all_distances:
            rand_triplets = sample_random_triplets(
                pairwise_dist_matrix, n_random, sig, distance_index
            )
        else:
            rand_triplets = sample_random_triplets(X, n_random, sig, distance_index)
        rand_weights = rand_triplets[:, -1]
        rand_triplets = rand_triplets[:, :-1].astype(np.int32)
        triplets = np.vstack((triplets, rand_triplets))
        weights = np.hstack((weights, rand_weights))
    weights[np.isnan(weights)] = 0.0
    weights -= np.min(weights)
    weights = tempered_log(1.0 + weights, weight_temp)
    return (triplets, weights)


@njit("void(f4[:,:],f4[:,:],f4[:,:],f4,i4,i4)", parallel=True, nogil=True)
def update_embedding(Y, grad, vel, lr, iter_num, opt_method):
    n, dim = Y.shape
    if opt_method == 0:  # sd
        for i in prange(n):
            for d in prange(dim):
                Y[i][d] -= lr * grad[i][d]
    elif opt_method == 1:  # momentum
        gamma = _FINAL_MOMENTUM if iter_num > _SWITCH_ITER_MOMENTUM else _INIT_MOMENTUM
        for i in prange(n):
            for d in prange(dim):
                vel[i][d] = gamma * vel[i][d] - lr * grad[i][d]  # - 1e-5 * Y[i,d]
                Y[i][d] += vel[i][d]


@njit("void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,i4)", parallel=True, nogil=True)
def update_embedding_dbd(Y, grad, vel, gain, lr, iter_num):
    n, dim = Y.shape
    # momentum parameter
    gamma = _FINAL_MOMENTUM if iter_num > _SWITCH_ITER_MOMENTUM else _INIT_MOMENTUM
    for i in prange(n):
        for d in prange(dim):
            gain[i][d] = (
                (gain[i][d] + _INCREASE_GAIN)
                if (np.sign(vel[i][d]) != np.sign(grad[i][d]))
                else np.maximum(gain[i][d] * _DAMP_GAIN, _MIN_GAIN)
            )
            vel[i][d] = gamma * vel[i][d] - lr * gain[i][d] * grad[i][d]
            Y[i][d] += vel[i][d]


@njit("f4[:,:](f4[:,:],i4,i4,i4[:,:],f4[:])", parallel=True, nogil=True)
def trimap_grad(Y, n_inliers, n_outliers, triplets, weights):
    n, dim = Y.shape
    n_triplets = triplets.shape[0]
    grad = np.zeros((n + 1, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    y_ik = np.empty(dim, dtype=np.float32)
    n_viol = 0.0
    loss = 0.0
    n_knn_triplets = n * n_inliers * n_outliers
    for t in range(n_triplets):
        i = triplets[t, 0]
        j = triplets[t, 1]
        k = triplets[t, 2]
        if (t % n_outliers) == 0 or (
            t >= n_knn_triplets
        ):  # update y_ij, y_ik, d_ij, d_ik
            d_ij = 1.0
            d_ik = 1.0
            for d in range(dim):
                y_ij[d] = Y[i, d] - Y[j, d]
                y_ik[d] = Y[i, d] - Y[k, d]
                d_ij += y_ij[d] ** 2
                d_ik += y_ik[d] ** 2
        else:  # update y_ik and d_ik only
            d_ik = 1.0
            for d in range(dim):
                y_ik[d] = Y[i, d] - Y[k, d]
                d_ik += y_ik[d] ** 2
        if d_ij > d_ik:
            n_viol += 1.0
        loss += weights[t] * 1.0 / (1.0 + d_ik / d_ij)
        w = weights[t] / (d_ij + d_ik) ** 2
        for d in range(dim):
            gs = y_ij[d] * d_ik * w
            go = y_ik[d] * d_ij * w
            grad[i, d] += gs - go
            grad[j, d] -= gs
            grad[k, d] += go
    grad[-1, 0] = loss
    grad[-1, 1] = n_viol
    return grad


def trimap(
    X,
    triplets,
    weights,
    knn_tuple,
    use_dist_matrix,
    n_dims,
    n_inliers,
    n_outliers,
    n_random,
    distance,
    lr,
    n_iters,
    Yinit,
    weight_temp,
    apply_pca,
    opt_method,
    verbose,
    return_seq,
):
    """Apply TriMap."""

    opt_method_dict = {"sd": 0, "momentum": 1, "dbd": 2}
    if verbose:
        t = time.time()
    n, dim = X.shape
    assert n_inliers < n - 1, "n_inliers must be less than (number of data points - 1)."
    if verbose:
        print("running TriMap on %d points with dimension %d" % (n, dim))
    pca_solution = False
    if triplets is None:
        if knn_tuple is not None:
            if verbose:
                print("using pre-computed knn")
            knn_nbrs, knn_distances = knn_tuple
            knn_nbrs = knn_nbrs.astype(np.int32)
            knn_distances = knn_distances.astype(np.float32)
            triplets, weights = generate_triplets_known_knn(
                X,
                knn_nbrs,
                knn_distances,
                n_inliers,
                n_outliers,
                n_random,
                None,
                distance,
                verbose,
                weight_temp,
            )
        elif use_dist_matrix:
            if verbose:
                print("using distance matrix")
            pairwise_dist_matrix = X
            pairwise_dist_matrix = pairwise_dist_matrix.astype(np.float32)
            n_extra = min(n_inliers + 50, n)
            knn_nbrs = np.zeros((n, n_extra), dtype=np.int32)
            knn_distances = np.zeros((n, n_extra), dtype=np.float32)
            for nn in range(n):
                bottom_k_indices = np.argpartition(
                    pairwise_dist_matrix[nn, :], n_extra
                )[:n_extra]
                bottom_k_distances = pairwise_dist_matrix[nn, bottom_k_indices]
                sort_indices = np.argsort(bottom_k_distances)
                knn_nbrs[nn, :] = bottom_k_indices[sort_indices]
                knn_distances[nn, :] = bottom_k_distances[sort_indices]
            triplets, weights = generate_triplets_known_knn(
                X,
                knn_nbrs,
                knn_distances,
                n_inliers,
                n_outliers,
                n_random,
                pairwise_dist_matrix,
                distance,
                verbose,
                weight_temp,
            )
        else:
            if verbose:
                print("pre-processing")
            if distance != "hamming":
                if dim > _PCA_DIM and apply_pca:
                    X -= np.mean(X, axis=0)
                    X = TruncatedSVD(
                        n_components=_PCA_DIM, random_state=0
                    ).fit_transform(X)
                    pca_solution = True
                    if verbose:
                        print("applied PCA")
                else:
                    X -= np.min(X)
                    X /= np.max(X)
                    X -= np.mean(X, axis=0)
            triplets, weights = generate_triplets(
                X, n_inliers, n_outliers, n_random, distance, verbose, weight_temp
            )
            if verbose:
                print("sampled triplets")
    else:
        if verbose:
            print("using stored triplets")

    if Yinit == "random":
        Y = np.random.normal(size=[n, n_dims]).astype(np.float32) * _INIT_RAND_SCALE
    elif not Yinit or Yinit == "pca":
        if pca_solution:
            Y = _INIT_PCA_SCALE * X[:, :n_dims]
        else:
            Y = _INIT_PCA_SCALE * PCA(n_components=n_dims).fit_transform(X).astype(
                np.float32
            )
    else:
        Y = Yinit.astype(np.float32)
    if return_seq:
        Y_all = np.zeros((n, n_dims, int(n_iters / _RETURN_EVERY + 1)))
        Y_all[:, :, 0] = Yinit

    C = np.inf
    tol = 1e-7
    n_triplets = float(triplets.shape[0])
    if verbose:
        print("running TriMap with " + opt_method)
    vel = np.zeros_like(Y, dtype=np.float32)
    if opt_method_dict[opt_method] == 2:
        gain = np.ones_like(Y, dtype=np.float32)

    for itr in range(n_iters):
        old_C = C
        if opt_method_dict[opt_method] == 0:
            grad = trimap_grad(Y, n_inliers, n_outliers, triplets, weights)
        else:
            # momentum parameter
            gamma = _FINAL_MOMENTUM if itr > _SWITCH_ITER_MOMENTUM else _INIT_MOMENTUM
            grad = trimap_grad(
                Y + gamma * vel, n_inliers, n_outliers, triplets, weights
            )
        C = grad[-1, 0] / n_triplets
        n_viol = grad[-1, 1]

        # update Y
        if opt_method_dict[opt_method] < 2:
            update_embedding(Y, grad, vel, lr, itr, opt_method_dict[opt_method])
        else:
            update_embedding_dbd(Y, grad, vel, gain, lr, itr)

        # update the learning rate
        if opt_method_dict[opt_method] < 2:
            lr = lr * 1.01 if old_C > C + tol else lr * 0.9
        if return_seq and (itr + 1) % _RETURN_EVERY == 0:
            Y_all[:, :, int((itr + 1) / _RETURN_EVERY)] = Y
        if verbose:
            if (itr + 1) % _DISP_ITER == 0:
                print(
                    "Iteration: %4d / %4d, Loss: %3.3f, Violated triplets: %0.4f"
                    % (itr + 1, n_iters, C, n_viol / n_triplets * 100.0)
                )
    if verbose:
        elapsed = str(datetime.timedelta(seconds=time.time() - t))
        print("Elapsed time: %s" % (elapsed))
    if return_seq:
        return (Y_all, triplets, weights)
    else:
        return (Y, triplets, weights)


class TRIMAP(BaseEstimator):
    """TriMap: Dimensionality Reduction Using Triplet Constraints.

    Find a low-dimensional representation of the data by satisfying the sampled
    triplet constraints from the high-dimensional features.

    Input
    ------

    n_dims: Number of dimensions of the embedding (default = 2).

    n_inliers: Number of inlier points for triplet constraints (default = 12).

    n_outliers: Number of outlier points for triplet constraints (default = 4).

    n_random: Number of random triplet constraints per point (default = 3).

    distance: Distance measure ('euclidean' (default), 'manhattan', 'angular' (or 'cosine'),
    'hamming').

    lr: Learning rate (default = 0.1).

    n_iters: Number of iterations (default = 400).

    triplets: Use pre-computed triplets (default = None).

    weights: Use pre-computed weights (default = None).

    use_dist_matrix: If True, X is the pairwise distances between points (default = False).

    knn_tuple: Use the pre-computed nearest-neighbors information in form of a
    tuple (knn_nbrs, knn_distances), needs also X to compute the embedding (default = None).

    verbose: Print the progress report (default = False).

    weight_temp: Temperature of the logarithm applied to the weights. Larger temperatures generate
    more compact embeddings. weight_temp=0. corresponds to no transformation (default=0.5).

    apply_pca: Apply PCA to reduce the dimensions to 100 if necessary before the
    nearest-neighbor calculation (default = True).

    opt_method: Optimization method ('sd': steepest descent,  'momentum': GD
    with momentum, 'dbd': GD with momentum delta-bar-delta (default)).

    return_seq: Return the sequence of maps recorded every _RETURN_EVERY=10 iterations
    (default = False).
    """

    def __init__(
        self,
        n_dims=2,
        n_inliers=12,
        n_outliers=4,
        n_random=3,
        distance="euclidean",
        lr=0.1,
        n_iters=400,
        triplets=None,
        weights=None,
        use_dist_matrix=False,
        knn_tuple=None,
        verbose=False,
        weight_adj=None,
        weight_temp=0.5,
        apply_pca=True,
        opt_method="dbd",
        return_seq=False,
    ):
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
        self.verbose = verbose
        self.return_seq = return_seq

        if self.n_dims < 2:
            raise ValueError("The number of output dimensions must be at least 2.")
        if self.n_inliers < 1:
            raise ValueError("The number of inliers must be a positive number.")
        if self.n_outliers < 1:
            raise ValueError("The number of outliers must be a positive number.")
        if self.n_random < 0:
            raise ValueError(
                "The number of random triplets must be a non-negative number."
            )
        if self.lr <= 0:
            raise ValueError("The learning rate must be a positive value.")
        if self.use_dist_matrix:
            distance = "pre-computed"
            self.distance = distance
        if self.distance == "hamming" and apply_pca:
            warnings.warn("apply_pca = True for Hamming distance.")
        if self.distance == "cosine":
            self.distance = "angular"
        if weight_adj is not None:
            warnings.warn(
                "'weight_adj' is deprecated and will not be applied."
                " Adjust 'weight_temp' if needed."
            )

        if self.verbose:
            print(
                "TRIMAP(n_inliers={}, n_outliers={}, n_random={}, distance={}, weight_temp={}, "
                "lr={}, n_iters={}, apply_pca={}, opt_method={}, verbose={}, return_seq={})".format(
                    n_inliers,
                    n_outliers,
                    n_random,
                    distance,
                    weight_temp,
                    lr,
                    n_iters,
                    apply_pca,
                    opt_method,
                    verbose,
                    return_seq,
                )
            )
            if not self.apply_pca:
                print(
                    _BOLD
                    + "running ANNOY on high-dimensional data. nearest-neighbor search may be slow!"
                    + _RESET
                )

    def fit(self, X, init=None):
        """Runs TriMap on the input data X.

        Input
        ------

        X: Instance matrix

        init: Initial solution
        """
        X = X.astype(np.float32)

        self.embedding_, self.triplets, self.weights = trimap(
            X,
            self.triplets,
            self.weights,
            self.knn_tuple,
            self.use_dist_matrix,
            self.n_dims,
            self.n_inliers,
            self.n_outliers,
            self.n_random,
            self.distance,
            self.lr,
            self.n_iters,
            init,
            self.weight_temp,
            self.apply_pca,
            self.opt_method,
            self.verbose,
            self.return_seq,
        )
        return self

    def fit_transform(self, X, init=None):
        """
        Runs the TriMap algorithm on the input data X and returns the embedding

        Input
        ------

        X: Instance matrix

        init: Initial solution
        """
        self.fit(X, init)
        return self.embedding_

    def sample_triplets(self, X):
        """
        Samples and stores triplets

        Input
        ------

        X: Instance matrix
        """
        if self.verbose:
            print("pre-processing")
        X = X.astype(np.float32)
        if self.distance != "hamming":
            if X.shape[1] > _PCA_DIM and self.apply_pca:
                X -= np.mean(X, axis=0)
                X = TruncatedSVD(n_components=_PCA_DIM, random_state=0).fit_transform(X)
                if self.verbose:
                    print("applied PCA")
            else:
                X -= np.min(X)
                X /= np.max(X)
                X -= np.mean(X, axis=0)
        self.triplets, self.weights = generate_triplets(
            X,
            self.n_inliers,
            self.n_outliers,
            self.n_random,
            self.distance,
            self.verbose,
            self.weight_temp,
        )
        if self.verbose:
            print("sampled triplets")

        return self

    def del_triplets(self):
        """
        Deletes the stored triplets
        """
        self.triplets = None
        self.weights = None

        return self

    def global_score(self, X, Y):
        """
        Global score

        Input
        ------

        X: Instance matrix
        Y: Embedding
        """

        def global_loss_(X, Y):
            X = X - np.mean(X, axis=0)
            Y = Y - np.mean(Y, axis=0)
            A = X.T @ (Y @ np.linalg.inv(Y.T @ Y))
            return np.mean(np.power(X.T - A @ Y.T, 2))

        n_dims = Y.shape[1]
        Y_pca = PCA(n_components=n_dims).fit_transform(X)
        gs_pca = global_loss_(X, Y_pca)
        gs_emb = global_loss_(X, Y)
        return np.exp(-(gs_emb - gs_pca) / gs_pca)
