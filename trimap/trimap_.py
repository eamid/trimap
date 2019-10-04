#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

TriMap: Large-scale Dimensionality Reduction Using Triplet Constraints

"""


from sklearn.base import BaseEstimator
import numba
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import numpy as np
import time
import datetime
import sys
import warnings

if sys.version_info < (3,):
    range = xrange

bold = "\033[1m"
reset = "\033[0;0m"


@numba.njit('f4(f4[:])')
def l2_norm(x):
    """
    L2 norm of a vector.

    """
    result = 0.0
    for i in range(x.shape[0]):
        result += x[i]**2
    return np.sqrt(result)


@numba.njit('f4(f4[:],f4[:])')
def euclid_dist(x1, x2):
    """
    Euclidean distance between two vectors.

    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i]-x2[i])**2
    return np.sqrt(result)


@numba.njit('f4(f4[:],f4[:])')
def manhattan_dist(x1, x2):
    """
    Manhattan distance between two vectors.

    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += np.abs(x1[i]-x2[i])
    return result


@numba.njit('f4(f4[:],f4[:])')
def angular_dist(x1, x2):
    """
    Angular (i.e. cosine) distance between two vectors.

    """
    x1_norm = np.maximum(l2_norm(x1), 1e-20)
    x2_norm = np.maximum(l2_norm(x2), 1e-20)
    result = 0.0
    for i in range(x1.shape[0]):
        result += x1[i] * x2[i]
    return np.sqrt(2.0 - 2.0 * result/x1_norm/x2_norm)


@numba.njit('f4(f4[:],f4[:])')
def hamming_dist(x1, x2):
    """
    Hamming distance between two vectors.

    """
    result = 0.0
    for i in range(x1.shape[0]):
        if x1[i] != x2[i]:
            result += 1.0
    return result


@numba.njit()
def calculate_dist(x1, x2, distance_index):
    if distance_index == 0:
        return euclid_dist(x1, x2)
    elif distance_index == 1:
        return manhattan_dist(x1, x2)
    elif distance_index == 2:
        return angular_dist(x1, x2)
    elif distance_index == 3:
        return hamming_dist(x1, x2)


@numba.njit()
def rejection_sample(n_samples, max_int, rejects):
    """
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


@numba.njit('i4[:,:](f4[:,:],i4[:,:],i4,i4)', parallel=True, nogil=True)
def sample_knn_triplets(P, nbrs, n_inlier, n_outlier):
    """
    Sample nearest neighbors triplets based on the similarity values given in P

    Input
    ------

    nbrs: Nearest neighbors indices for each point. The similarity values 
        are given in matrix P. Row i corresponds to the i-th point.

    P: Matrix of pairwise similarities between each point and its neighbors 
        given in matrix nbrs

    n_inlier: Number of inlier points

    n_outlier: Number of outlier points

    Output
    ------

    triplets: Sampled triplets
    """
    n, n_neighbors = nbrs.shape
    triplets = np.empty((n * n_inlier * n_outlier, 3), dtype=np.int32)
    for i in numba.prange(n):
        sort_indices = np.argsort(-P[i])
        for j in numba.prange(n_inlier):
            sim = nbrs[i][sort_indices[j+1]]
            samples = rejection_sample(n_outlier, n, sort_indices[:j+2])
            for k in numba.prange(n_outlier):
                index = i * n_inlier * n_outlier + j * n_outlier + k
                out = samples[k]
                triplets[index][0] = i
                triplets[index][1] = sim
                triplets[index][2] = out
    return triplets



@numba.njit('f4[:,:](f4[:,:],i4,f4[:],i4)', parallel=True, nogil=True)
def sample_random_triplets(X, n_random, sig, distance_index):
    """
    Sample uniformly random triplets

    Input
    ------

    X: Instance matrix

    n_random: Number of random triplets per point

    sig: Scaling factor for the distances

    distance_index: index of the distance measure

    Output
    ------

    rand_triplets: Sampled triplets
    """
    n = X.shape[0]
    rand_triplets = np.empty((n * n_random, 4), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n_random):
            sim = np.random.choice(n)
            while sim == i:
                sim = np.random.choice(n)
            out = np.random.choice(n)
            while out == i or out == sim:
                out = np.random.choice(n)
            p_sim = np.exp(-calculate_dist(X[i],X[sim], distance_index)**2/(sig[i] * sig[sim]))
            if p_sim < 1e-20:
                p_sim = 1e-20
            p_out = np.exp(-calculate_dist(X[i],X[out], distance_index)**2/(sig[i] * sig[out]))
            if p_out < 1e-20:
                p_out = 1e-20
            if p_sim < p_out:
                sim, out = out, sim
                p_sim, p_out = p_out, p_sim
            rand_triplets[i * n_random + j][0] = i
            rand_triplets[i * n_random + j][1] = sim
            rand_triplets[i * n_random + j][2] = out
            rand_triplets[i * n_random + j][3] = p_sim/p_out
    return rand_triplets


@numba.njit('f4[:,:](f4[:,:],f4[:],i4[:,:])', parallel=True, nogil=True)
def find_p(distances, sig, nbrs):
    """
    Calculates the similarity matrix P

    Input
    ------

    distances: Matrix of pairwise distances

    sig: Scaling factor for the distances

    nbrs: Nearest neighbors

    Output
    ------

    P: Pairwise similarity matrix
    """
    n, n_neighbors = distances.shape
    P = np.zeros((n,n_neighbors), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n_neighbors):
            P[i][j] = np.exp(-distances[i][j]**2/sig[i]/sig[nbrs[i][j]])
    return P


@numba.njit('f4[:](i4[:,:],f4[:,:],i4[:,:],f4[:],f4[:])', parallel=True, nogil=True)
def find_weights(triplets, P, nbrs, distances, sig):
    """
    Calculates the weights for the sampled nearest neighbors triplets

    Input
    ------

    triplets: Sampled triplets

    P: Pairwise similarity matrix

    nbrs: Nearest neighbors

    distances: Matrix of pairwise distances

    sig: Scaling factor for the distances

    Output
    ------

    weights: Weights for the triplets
    """
    n_triplets = triplets.shape[0]
    weights = np.empty(n_triplets, dtype=np.float32)
    for t in numba.prange(n_triplets):
        i = triplets[t][0]
        sim = 0
        while(nbrs[i][sim] != triplets[t][1]):
            sim += 1
        p_sim = P[i][sim]
        p_out = np.exp(-distances[t]**2/(sig[i] * sig[triplets[t][2]]))
        if p_out < 1e-20:
            p_out = 1e-20
        weights[t] = p_sim/p_out
    return weights

def generate_triplets(X, n_inlier, n_outlier, n_random, distance='euclidean', apply_pca=True, weight_adj = True, verbose = True):
    distance_dict = {'euclidean':0, 'manhattan':1, 'angular':2, 'hamming':3}
    distance_index = distance_dict[distance]
    n, dim = X.shape
    if dim > 100 and apply_pca:
        X = TruncatedSVD(n_components=100, random_state=0).fit_transform(X)
        dim = 100
        if verbose:
            print("applied PCA")
    n_extra = min(max(n_inlier, 50),n)
    # n_extra = n_inlier + 1
    tree = AnnoyIndex(dim, metric=distance)
    for i in range(n):
        tree.add_item(i, X[i,:])
    tree.build(20)
    nbrs = np.empty((n,n_extra), dtype=np.int32)
    distances = np.empty((n,n_extra), dtype=np.float32)
    dij = np.empty(n_extra, dtype=np.float32)
    for i in range(n):
        nbrs[i,:] = tree.get_nns_by_item(i, n_extra)
        for j in range(n_extra):
            distances[i,j] = tree.get_distance(i, nbrs[i,j])
        sort_indices = np.argsort(distances[i,:])
        nbrs[i,:] = nbrs[i,sort_indices]
        distances[i,:] = distances[i,sort_indices]
    if verbose:
        print("found nearest neighbors")
    sig = np.maximum(np.mean(distances[:, 3:6], axis=1), 1e-10) # scale parameter
    P = find_p(distances, sig, nbrs)
    triplets = sample_knn_triplets(P, nbrs, n_inlier, n_outlier)
    n_triplets = triplets.shape[0]
    outlier_dist = np.empty(n_triplets, dtype=np.float32)
    for t in range(n_triplets):
        outlier_dist[t] = calculate_dist(X[triplets[t,0],:], X[triplets[t,2],:], distance_index)
    weights = find_weights(triplets, P, nbrs, outlier_dist, sig)
    if n_random > 0:
        rand_triplets = sample_random_triplets(X, n_random, sig, distance_index)
        rand_weights = rand_triplets[:,-1]
        rand_triplets = rand_triplets[:,:-1].astype(np.int32)
        triplets = np.vstack((triplets, rand_triplets))
        weights = np.hstack((weights, rand_weights))
    weights[np.isnan(weights)] = 0.0
    weights /= np.max(weights)
    weights += 0.0001
    if weight_adj:
        if not isinstance(weight_adj, (int, float)):
            weight_adj = 500.0
        weights = np.log(1 + weight_adj * weights)
        weights /= np.max(weights)
    return (triplets, weights)


@numba.njit('void(f4[:,:],f4[:,:],f4[:,:],f4,i4,i4)', parallel=True, nogil=True)
def update_embedding(Y, grad, vel, lr, iter_num, opt_method):
    n, dim = Y.shape
    if opt_method == 0: # sd
        for i in numba.prange(n):
            for d in numba.prange(dim):
                Y[i][d] -= lr * grad[i][d]
    elif opt_method == 1: # momentum
        if iter_num > 250:
            gamma = 0.5
        else:
            gamma = 0.3
        for i in numba.prange(n):
            for d in numba.prange(dim):
                vel[i][d] = gamma * vel[i][d] - lr * grad[i][d] # - 1e-5 * Y[i,d]
                Y[i][d] += vel[i][d]

@numba.njit('void(f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4,i4)', parallel=True, nogil=True)
def update_embedding_dbd(Y, grad, vel, gain, lr, iter_num):
    n, dim = Y.shape
    if  iter_num > 250:
        gamma = 0.8 # moment parameter
    else:
        gamma = 0.5
    min_gain = 0.01
    for i in numba.prange(n):
        for d in numba.prange(dim):
            gain[i][d] = (gain[i][d]+0.2) if (np.sign(vel[i][d]) != np.sign(grad[i][d])) else np.maximum(gain[i][d]*0.8, min_gain)
            vel[i][d] = gamma * vel[i][d] - lr * gain[i][d] * grad[i][d]
            Y[i][d] += vel[i][d]

@numba.njit('f4[:,:](f4[:,:],i4,i4,i4[:,:],f4[:])', parallel=True, nogil=True)
def trimap_grad(Y, n_inlier, n_outlier, triplets, weights):
    n, dim = Y.shape
    n_triplets = triplets.shape[0]
    grad = np.zeros((n, dim), dtype=np.float32)
    y_ij = np.empty(dim, dtype=np.float32)
    y_ik = np.empty(dim, dtype=np.float32)
    n_viol = 0.0
    loss = 0.0
    n_knn_triplets = n * n_inlier * n_outlier
    for t in range(n_triplets):
        i = triplets[t,0]
        j = triplets[t,1]
        k = triplets[t,2]
        if (t % n_outlier) == 0 or (t >= n_knn_triplets):  # update y_ij, y_ik, d_ij, d_ik
            d_ij = 1.0
            d_ik = 1.0
            for d in range(dim):
                y_ij[d] = Y[i,d] - Y[j,d]
                y_ik[d] = Y[i,d] - Y[k,d]
                d_ij += y_ij[d]**2
                d_ik += y_ik[d]**2
        else:  # update y_ik and d_ik only
            d_ik = 1.0
            for d in range(dim):
                y_ik[d] = Y[i,d] - Y[k,d]
                d_ik += y_ik[d]**2
        if (d_ij > d_ik):
            n_viol += 1.0
        loss += weights[t] * 1.0/(1.0 + d_ik/d_ij)
        w = weights[t]/(d_ij + d_ik)**2
        for d in range(dim):
            gs = y_ij[d] * d_ik * w
            go = y_ik[d] * d_ij * w
            grad[i,d] += gs - go
            grad[j,d] -= gs
            grad[k,d] += go
    last = np.zeros((1,dim), dtype=np.float32)
    last[0] = loss
    last[1] = n_viol
    return np.vstack((grad, last))  
    
def trimap(X, triplets, weights, n_dims, n_inliers, n_outliers, n_random, distance, lr, n_iters, Yinit,
 weight_adj, apply_pca, opt_method, verbose, return_seq):
    """
    Apply TriMap.

    """

    opt_method_dict = {'sd':0, 'momentum':1, 'dbd':2}
    if verbose:
        t = time.time()
    n, dim = X.shape
    if verbose:
        print("running TriMap on %d points with dimension %d" % (n, dim))
    if triplets[0] is None:
        if verbose:
            print("pre-processing")
        if distance != 'hamming':
            X -= np.min(X)
            X /= np.max(X)
            X -= np.mean(X,axis=0)
        triplets, weights = generate_triplets(X, n_inliers, n_outliers, n_random, distance, apply_pca, weight_adj, verbose)
        if verbose:
            print("sampled triplets")
    else:
        if verbose:
            print("using stored triplets")

    if Yinit is None or Yinit is 'pca':
        Y = 0.01 * PCA(n_components = n_dims).fit_transform(X).astype(np.float32)
    elif Yinit is 'random':
        Y = np.random.normal(size=[n, n_dims]).astype(np.float32) * 0.0001
    else:
        Y = Yinit.astype(np.float32)
    if return_seq:
        Y_all = np.zeros((n, n_dims, int(n_iters/10 + 1)))
        Y_all[:,:,0] = Yinit

    C = np.inf
    tol = 1e-7
    n_triplets = float(triplets.shape[0])
    lr = lr * n / n_triplets
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
            if itr > 250:
                gamma = 0.5
            else:
                gamma = 0.3
            grad = trimap_grad(Y + gamma * vel, n_inliers, n_outliers, triplets, weights)
        C = grad[-1,0]
        n_viol = grad[-1,1]
            
        # update Y
        if opt_method_dict[opt_method] < 2:
            update_embedding(Y, grad, vel, lr, itr, opt_method_dict[opt_method])
        else:
            update_embedding_dbd(Y, grad, vel, gain, lr, itr)
        
        # update the learning rate
        if opt_method_dict[opt_method] < 2:
            if old_C > C + tol:
                lr = lr * 1.01
            else:
                lr = lr * 0.9
        if return_seq and (itr+1) % 10 == 0:
            Y_all[:,:,int((itr+1)/10)] = Y
        if verbose:
            if (itr+1) % 100 == 0:
                print('Iteration: %4d, Loss: %3.3f, Violated triplets: %0.4f' % (itr+1, C, n_viol/n_triplets*100.0))
    if verbose:
        elapsed = str(datetime.timedelta(seconds= time.time() - t))
        print("Elapsed time: %s" % (elapsed))
    if return_seq:
        return (Y_all, triplets, weights)
    else:
        return (Y, triplets, weights)

class TRIMAP(BaseEstimator):
    """
    Dimensionality Reduction Using Triplet Constraints

    Find a low-dimensional repersentation of the data by satisfying the sampled
    triplet constraints from the high-dimensional features.

    Input
    ------

    n_dims: Number of dimensions of the embedding (default = 2)

    n_inliers: Number of inlier points for triplet constraints (default = 10)

    n_outliers: Number of outlier points for triplet constraints (default = 5)

    n_random: Number of random triplet constraints per point (default = 5)

    distance: Distance measure ('euclidean' (default), 'manhattan', 'angular', 'hamming')

    lr: Learning rate (default = 1000.0)

    n_iters: Number of iterations (default = 400)

    apply_pca: Apply PCA to reduce the dimensions to 100 if necessary before the nearest-neighbor calculation (default = True)

    opt_method: Optimization method ('sd': steepest descent,  'momentum': GD with momentum, 'dbd': GD with momentum delta-bar-delta (default))

    verbose: Print the progress report (default = True)

    weight_adj: Adjusting the weights using a non-linear transformation (default = 500.0)

    return_seq: Return the sequence of maps recorded every 10 iterations (default = False)
    """

    def __init__(self,
                 n_dims=2,
                 n_inliers=10,
                 n_outliers=5,
                 n_random=5,
                 distance='euclidean',
                 lr=1000.0,
                 n_iters=400,
                 triplets=None,
                 weights=None,
                 verbose=True,
                 weight_adj=500.0,
                 apply_pca=True,
                 opt_method='dbd',
                 return_seq=False
                 ):
        self.n_dims = n_dims
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.distance = distance
        self.lr = lr
        self.n_iters = n_iters
        self.triplets = triplets,
        self.weights = weights
        self.weight_adj = weight_adj
        self.apply_pca = apply_pca
        self.opt_method = opt_method
        self.verbose = verbose
        self.return_seq = return_seq

        if self.n_dims < 2:
            raise ValueError('The number of output dimensions must be at least 2.')
        if self.n_inliers < 1:
            raise ValueError('The number of inliers must be a positive number.')
        if self.n_outliers < 1:
            raise ValueError('The number of outliers must be a positive number.')
        if self.n_random < 0:
            raise ValueError('The number of random triplets must be a non-negative number.')
        if self.lr <= 0:
            raise ValueError('The learning rate must be a positive value.')
        if self.distance == 'hamming' and apply_pca:
            warnings.warn('apply_pca = True for Hamming distance.')
            
        if self.verbose:
            print("TRIMAP(n_inliers={}, n_outliers={}, n_random={}, distance={},"
                  "lr={}, n_iters={}, weight_adj={}, apply_pca={}, opt_method={}, verbose={}, return_seq={})".format(
                  n_inliers, n_outliers, n_random, distance, lr, n_iters, weight_adj, apply_pca, opt_method, verbose, return_seq))
            if not self.apply_pca:
                print(bold + "running ANNOY on high-dimensional data. nearest-neighbor search may be slow!" + reset)

    def fit(self, X, init = None):
        """
        Runs the TriMap algorithm on the input data X

        Input
        ------

        X: Instance matrix

        init: Initial solution
        """
        X = X.astype(np.float32)
        
        self.embedding_, self.triplets, self.weights = trimap(X, self.triplets,
            self.weights, self.n_dims, self.n_inliers, self.n_outliers, self.n_random, self.distance,
            self.lr, self.n_iters, init, self.weight_adj, self.apply_pca, self.opt_method, self.verbose, self.return_seq)
        return self

    def fit_transform(self, X, init = None):
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
        if self.distance != 'hamming':
            X -= np.min(X)
            X /= np.max(X)
            X -= np.mean(X,axis=0)
        self.triplets, self.weights = generate_triplets(X, self.n_inliers, self.n_outliers, self.n_random, self.distance, self.apply_pca, self.weight_adj, self.verbose)
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
        Y_pca = PCA(n_components = n_dims).fit_transform(X)
        gs_pca = global_loss_(X, Y_pca)
        gs_emb = global_loss_(X, Y)
        return np.exp(-(gs_emb-gs_pca)/gs_pca)
        

