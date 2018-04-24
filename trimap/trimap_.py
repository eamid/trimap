#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

TriMap: Dimensionality Reduction Using Triplet Constraints

@author: Ehsan Amid
"""


from sklearn.base import BaseEstimator
import numba
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.decomposition import TruncatedSVD
import numpy as np
import time


@numba.njit()
def euclid_dist(x1, x2):
    """
    Fast Euclidean distance calculation between two vectors.

    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i]-x2[i])**2
    return np.sqrt(result)

@numba.njit()
def rejection_sample(n_samples, max_int, rejects):
    """
    Samples "n_samples" integers from a given interval [0,max_int] while
    rejecting the values that are in the "rejects".

    """
    result = np.empty(n_samples, dtype=np.int64)
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


@numba.njit('i8[:,:](f8[:,:],i8[:,:], i8,i8)', parallel=True, nogil=True)
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
    triplets = np.empty((n * n_inlier * n_outlier, 3), dtype=np.int64)
    for i in xrange(n):
        sort_indices = np.argsort(-P[i,:])
        for j in range(n_inlier):
            sim = nbrs[i,sort_indices[j+1]]
            samples = rejection_sample(n_outlier, n, sort_indices[j+1:])
            for k in range(n_outlier):
                index = i * n_inlier * n_outlier + j * n_outlier + k
                out = samples[k]
                triplets[index,0] = i
                triplets[index,1] = sim
                triplets[index,2] = out
    return triplets



@numba.njit('f8[:,:](f8[:,:],i8,f8[:])', parallel=True, nogil=True)
def sample_random_triplets(X, n_random, sig):
    """
    Sample uniformly random triplets

    Input
    ------

    X: Instance matrix

    n_random: Number of random triplets per point

    sig: Scaling factor for the distances

    Output
    ------

    rand_triplets: Sampled triplets
    """
    n = X.shape[0]
    rand_triplets = np.empty((n * n_random, 4), dtype=np.float64)
    for i in xrange(n):
        for j in range(n_random):
            sim = np.random.choice(n)
            while sim == i:
                sim = np.random.choice(n)
            out = np.random.choice(n)
            while out == i or out == sim:
                out = np.random.choice(n)
            p_sim = np.exp(-euclid_dist(X[i,:],X[sim,:])**2/(sig[i] * sig[sim]))
            if p_sim < 1e-20:
                p_sim = 1e-20
            p_out = np.exp(-euclid_dist(X[i,:],X[out,:])**2/(sig[i] * sig[out]))
            if p_out < 1e-20:
                p_out = 1e-20
            if p_sim < p_out:
                sim, out = out, sim
                p_sim, p_out = p_out, p_sim
            rand_triplets[i * n_random + j,0] = i
            rand_triplets[i * n_random + j,1] = sim
            rand_triplets[i * n_random + j,2] = out
            rand_triplets[i * n_random + j,3] = p_sim/p_out
    return rand_triplets


@numba.njit('f8[:,:](f8[:,:],f8[:],i8[:,:])', parallel=True, nogil=True)
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
    P = np.zeros((n,n_neighbors), dtype=np.float64)
    for i in range(n):
        for j in range(n_neighbors):
            P[i,j] = np.exp(-distances[i,j]**2/sig[i]/sig[nbrs[i,j]])
    return P


@numba.njit('f8[:](i8[:,:],f8[:,:],i8[:,:],f8[:],f8[:])',parallel=True, nogil=True)
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
    weights = np.empty(n_triplets, dtype=np.float64)
    for t in xrange(n_triplets):
        i = triplets[t,0]
        sim = 0
        while(nbrs[i,sim] != triplets[t,1]):
            sim += 1
        p_sim = P[i,sim]
        p_out = np.exp(-distances[t]**2/(sig[i] * sig[triplets[t,2]]))
        if p_out < 1e-20:
            p_out = 1e-20
        weights[t] = p_sim/p_out
    return weights

def generate_triplets(X, n_inlier, n_outlier, n_random, verbose = True):
    n, dim = X.shape
    n_extra = max(n_inlier, 70)
    exact = n <= 1e4 or dim <= 50
    if exact: # do exact knn search
        if dim > 50:
            X = TruncatedSVD(n_components=50, random_state=0).fit_transform(X)
        knn_tree = knn(n_neighbors= n_extra, algorithm='auto').fit(X)
        distances, nbrs = knn_tree.kneighbors(X)
        distances = np.empty((n,n_extra), dtype=np.float64)
        for i in xrange(n):
            for j in xrange(n_extra):
                distances[i,j] = euclid_dist(X[i,:], X[nbrs[i,j],:])
    else: # use annoy
        tree = AnnoyIndex(dim)
        for i in xrange(n):
            tree.add_item(i, X[i,:])
        tree.build(10)
        nbrs = np.empty((n,n_extra), dtype=np.int64)
        distances = np.empty((n,n_extra), dtype=np.float64)
        for i in xrange(n):
            nbrs[i,:] = tree.get_nns_by_item(i, n_extra)
            for j in xrange(n_extra):
                distances[i,j] = tree.get_distance(i, nbrs[i,j])
    if verbose:
        print("found nearest neighbors")
    sig = np.maximum(np.mean(distances[:, 10:20], axis=1), 1e-20) # scale parameter
    P = find_p(distances, sig, nbrs)
    triplets = sample_knn_triplets(P, nbrs, n_inlier, n_outlier)
    n_triplets = triplets.shape[0]
    outlier_dist = np.empty(n_triplets, dtype=np.float64)
    if exact:
        for t in xrange(n_triplets):
            outlier_dist[t] = np.sqrt(np.sum((X[triplets[t,0],:] - X[triplets[t,2],:])**2))
    else:
        for t in xrange(n_triplets):
            outlier_dist[t] = tree.get_distance(triplets[t,0], triplets[t,2])
    weights = find_weights(triplets, P, nbrs, outlier_dist, sig)
    if n_random > 0:
        rand_triplets = sample_random_triplets(X, n_random, sig)
        rand_weights = rand_triplets[:,-1]
        rand_triplets = rand_triplets[:,:-1].astype(np.int64)
        triplets = np.vstack((triplets, rand_triplets))
        weights = np.hstack((weights, rand_weights))
    weights /= np.max(weights)
    weights += 0.0001
    return (triplets, weights)


@numba.njit('void(f8[:,:],f8[:,:],f8)', parallel=True, nogil=True)
def update_embedding(Y, grad, lr):
    n, dim = Y.shape
    for i in xrange(n):
        for d in xrange(dim):
            Y[i,d] -= lr * grad[i,d]
            
@numba.njit('f8[:,:](f8[:,:],i8,i8,i8[:,:],f8[:])', parallel=True, nogil=True)
def trimap_grad(Y, n_inlier, n_outlier, triplets, weights):
    n, dim = Y.shape
    n_triplets = triplets.shape[0]
    grad = np.zeros((n, dim), dtype=np.float64)
    y_ij = np.empty(dim, dtype=np.float64)
    y_ik = np.empty(dim, dtype=np.float64)
    n_viol = 0.0
    loss = 0.0
    n_knn_triplets = n * n_inlier * n_outlier
    for t in xrange(n_triplets):
        i = triplets[t,0]
        j = triplets[t,1]
        k = triplets[t,2]
        if (t % n_outlier) == 0 or (t >= n_knn_triplets): # update y_ij, d_ij
            d_ij = 1.0
            d_ik = 1.0
            for d in xrange(dim):
                y_ij[d] = Y[i,d] - Y[j,d]
                y_ik[d] = Y[i,d] - Y[k,d]
                d_ij += y_ij[d]**2
                d_ik += y_ik[d]**2
        else:
            d_ik = 1.0
            for d in xrange(dim):
                y_ik[d] = Y[i,d] - Y[k,d]
                d_ik += y_ik[d]**2
        if (d_ij > d_ik):
            n_viol += 1.0
        loss += weights[t] * 1.0/(1.0 + d_ik/d_ij)
        w = weights[t]/(d_ij + d_ik)**2
        for d in xrange(dim):
            gs = y_ij[d] * d_ik * w
            go = y_ik[d] * d_ij * w
            grad[i,d] += gs - go
            grad[j,d] -= gs
            grad[k,d] += go
    last = np.zeros((1,dim), dtype=np.float64)
    last[0] = loss
    last[1] = n_viol
    return np.vstack((grad, last))
    
    
def trimap(X, n_dims, n_inliers, n_outliers, n_random, lr, n_iters, Yinit, verbose):
    if verbose:
        t = time.time()
    n, dim = X.shape
    if verbose:
        print "running TriMap on %d points with dimension %d" % (n, dim)
        print "pre-processing"
    X -= np.min(X)
    X /= np.max(X)
    X -= np.mean(X,axis=0)
    triplets, weights = generate_triplets(X, n_inliers, n_outliers, n_random, verbose)
    if verbose:
        print("sampled triplets")
        
    if Yinit is None:
        Y = np.random.normal(size=[n, n_dims]) * 0.0001
    else:
        Y = Yinit      
    C = np.inf
    tol = 1e-7
    n_triplets = float(triplets.shape[0])
    
    if verbose:
        print("running TriMap")
    for itr in range(n_iters):
        old_C = C
        grad = trimap_grad(Y, n_inliers, n_outliers, triplets, weights)
        C = grad[-1,0]
        n_viol = grad[-1,1]
            
        # update Y
        update_embedding(Y, grad, lr * n/n_triplets)
        
        # update the learning rate
        if old_C > C + tol:
            lr = lr * 1.05
        else:
            lr = lr * 0.5
        
        if verbose:
            if (itr+1) % 100 == 0:
                print 'Iteration: %4d, Loss: %3.3f, Violated triplets: %0.4f' % (itr+1, C, n_viol/n_triplets*100.0)
    if verbose:
        print "Elapsed time %s" % (time.time() - t)
    return Y

class TRIMAP(BaseEstimator):
    """
    Dimensionality Reduction Using Triplet Constraints

    Find a low-dimensional repersentation of the data by satisfying the sampled
    triplet constraints from the high-dimensional features.

    Input
    ------

    n_dims: Number of dimensions of the embedding (default = 2)

    n_inliers: Number of inlier points for triplet constraints (default = 50)

    n_outliers: Number of outlier points for triplet constraints (default = 5)

    n_random: Number of random triplet constraints per point (default = 10)

    lr: Learning rate (default = 10000.0)

    n_iters: Number of iterations (default = 1500)

    verbose: Print the progress report (default = True)
    """

    def __init__(self,
                 n_dims=2,
                 n_inliers=50,
                 n_outliers=10,
                 n_random=5,
                 lr=10000.0,
                 n_iters = 1500,
                 verbose=True
                 ):
    	self.n_dims = n_dims
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.lr = lr
        self.n_inliers = n_inliers
        self.verbose = verbose

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

        if self.verbose:
            print("TRIMAP(n_inliers={}, n_outliers={}, n_random={}, "
                  "lr={}, n_iters={}, verbose={})".format(
                  n_inliers, n_outliers, n_random, lr, n_iters, verbose))

    def fit(self, X, init = None):
        """
        Runs the TriMap algorithm on the input data X

        Input
        ------

        X: Instance matrix

        init: Initial solution
        """
        X = X.astype(np.float64)
        
        self.embedding_ = trimap(X, self.n_dims, self.n_inliers, self.n_outliers, self.n_random, 
                                 self.lr, self.n_iters, init, self.verbose)
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
