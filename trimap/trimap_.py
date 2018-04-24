#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:44:46 2018

@author: Ehsan
"""


from sklearn.base import BaseEstimator
import numba
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.decomposition import TruncatedSVD
import numpy as np
import time
import sys


@numba.njit()
def euclid_dist(x1, x2):
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i]-x2[i])**2
    return np.sqrt(result)

@numba.njit()
def rejection_sample(n_samples, max_int, rejects):
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

def knn_annoy(X, num_neighbors = 50):
    n = X.shape[0]
    tree = AnnoyIndex(X.shape[1])
    for i in xrange(n):
        tree.add_item(i, X[i,:])
    tree.build(10)
    nbrs = np.empty((n,num_neighbors), dtype=np.int64)
    distances = np.empty((n,num_neighbors), dtype=np.float64)
    for i in xrange(n):
        nbrs[i,:] = tree.get_nns_by_item(i, num_neighbors)
        for j in xrange(num_neighbors):
            distances[i,j] = tree.get_distance(i, nbrs[i,j])
    return (nbrs, distances, tree)

@numba.njit('i8[:,:](f8[:,:],i8[:,:],f8[:,:], i8,i8)', parallel=True, nogil=True)
def sample_knn_triplets(P, nbrs, distances, kin, kout):
    n, num_neighbors = nbrs.shape
    triplets = np.empty((n * kin * kout, 3), dtype=np.int64)
    for i in xrange(n):
        sort_indices = np.argsort(-P[i,:])
        for j in range(kin):
            sim = nbrs[i,sort_indices[j+1]]
            samples = rejection_sample(kout, n, sort_indices[j+1:])
            for k in range(kout):
                index = i * kin * kout + j * kout + k
                out = samples[k]
                triplets[index,0] = i
                triplets[index,1] = sim
                triplets[index,2] = out
    return triplets

@numba.njit('f8[:,:](f8[:,:],i8,f8[:])', parallel=True, nogil=True)
def sample_random_triplets(X, krand, sig):
    n = X.shape[0]
    rand_triplets = np.empty((n * krand, 4), dtype=np.float64)
    for i in xrange(n):
        for j in range(krand):
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
            rand_triplets[i * krand + j,0] = i
            rand_triplets[i * krand + j,1] = sim
            rand_triplets[i * krand + j,2] = out
            rand_triplets[i * krand + j,3] = p_sim/p_out
    return rand_triplets

@numba.njit('f8[:,:](f8[:,:],f8[:],i8[:,:])', parallel=True, nogil=True)
def find_p(distances, sig, nbrs):
    n, num_neighbors = distances.shape
    P = np.zeros((n,num_neighbors), dtype=np.float64)
    for i in range(n):
        for j in range(num_neighbors):
            P[i,j] = np.exp(-distances[i,j]**2/sig[i]/sig[nbrs[i,j]])
    return P

@numba.njit('f8[:](i8[:,:],f8[:,:],i8[:,:],f8[:],f8[:])',parallel=True, nogil=True)
def find_weights(triplets, P, nbrs, distances, sig):
    num_triplets = triplets.shape[0]
    weights = np.empty(num_triplets, dtype=np.float64)
    for t in xrange(num_triplets):
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

def generate_triplets(X, kin, kout, krand, verbose = True):
    n, dim = X.shape
    num_neighbors = max(kin, 70)
    exact = n <= 1e4 or dim <= 50
    if exact: # do exact knn search
        if dim > 50:
            X = TruncatedSVD(n_components=50, random_state=0).fit_transform(X)
        knn_tree = knn(n_neighbors= num_neighbors, algorithm='auto').fit(X)
        distances, nbrs = knn_tree.kneighbors(X)
        distances = np.empty((n,num_neighbors), dtype=np.float64)
        for i in xrange(n):
            for j in xrange(num_neighbors):
                distances[i,j] = euclid_dist(X[i,:], X[nbrs[i,j],:])
    else: # use annoy
        tree = AnnoyIndex(dim)
        for i in xrange(n):
            tree.add_item(i, X[i,:])
        tree.build(10)
        nbrs = np.empty((n,num_neighbors), dtype=np.int64)
        distances = np.empty((n,num_neighbors), dtype=np.float64)
        for i in xrange(n):
            nbrs[i,:] = tree.get_nns_by_item(i, num_neighbors)
            for j in xrange(num_neighbors):
                distances[i,j] = tree.get_distance(i, nbrs[i,j])
    if verbose:
        print("found nearest neighbors")
    sig = np.maximum(np.mean(distances[:, 10:20], axis=1), 1e-20) # scale parameter
    P = find_p(distances, sig, nbrs)
    triplets = sample_knn_triplets(P, nbrs, distances, kin, kout)
    num_triplets = triplets.shape[0]
    outlier_dist = np.empty(num_triplets, dtype=np.float64)
    if exact:
        for t in xrange(num_triplets):
            outlier_dist[t] = np.sqrt(np.sum((X[triplets[t,0],:] - X[triplets[t,2],:])**2))
    else:
        for t in xrange(num_triplets):
            outlier_dist[t] = tree.get_distance(triplets[t,0], triplets[t,2])
    weights = find_weights(triplets, P, nbrs, outlier_dist, sig)
    if krand > 0:
        rand_triplets = sample_random_triplets(X, krand, sig)
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
def trimap_grad(Y, kin, kout, triplets, weights):
    n, dim = Y.shape
    num_triplets = triplets.shape[0]
    grad = np.zeros((n, dim), dtype=np.float64)
    y_ij = np.empty(dim, dtype=np.float64)
    y_ik = np.empty(dim, dtype=np.float64)
    num_viol = 0.0
    loss = 0.0
    num_knn_triplets = n * kin * kout
    for t in xrange(num_triplets):
        i = triplets[t,0]
        j = triplets[t,1]
        k = triplets[t,2]
        if (t % kout) == 0 or (t >= num_knn_triplets): # update y_ij, d_ij
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
            num_viol += 1.0
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
    last[1] = num_viol
    return np.vstack((grad, last))
    
    
def trimap(X, num_dims = 2, kin = 50, kout = 5, krand = 10, eta = 10000.0, num_iter = 1500, Yinit = None, verbose = True):
    if verbose:
        t = time.time()
    n, dim = X.shape
    if verbose:
        print "running TriMap on %d points with dimension %d" % (n, dim)
        print("pre-processing")
    X -= np.min(X)
    X /= np.max(X)
    X -= np.mean(X,axis=0)
    triplets, weights = generate_triplets(X, kin, kout, krand, verbose)
    if verbose:
        print("sampled triplets")
    
    if Yinit is None:
        Y = np.random.normal(size=[n, num_dims]) * 0.0001
    else:
        Y = Yinit
        
    C = np.inf
    tol = 1e-7
    num_triplets = float(triplets.shape[0])
    
    if verbose:
        print("running TriMap")
    for itr in range(num_iter):
        old_C = C
        grad = trimap_grad(Y, kin, kout, triplets, weights)
        C = grad[-1,0]
        num_viol = grad[-1,1]
            
        # update Y
        update_embedding(Y, grad, eta * n/num_triplets)
        
        # update the learning rate
        if old_C > C + tol:
            eta = eta * 1.05
        else:
            eta = eta * 0.5
        
        if verbose:
            if (itr+1) % 100 == 0:
                print 'Iteration: %4d, Loss: %3.3f, Violated triplets: %0.4f' % (itr+1, C, num_viol/num_triplets*100.0)
    if verbose:
        print "Elapsed time %s" % (time.time() - t)
    return Y

class TRIMAP(BaseEstimator):
    """
    TRIMAP
    """

    def __init__(self,
    			 n_dims=2,
                 n_neighbors=50,
                 n_outliers=5,
                 n_random=5,
                 lr=10000.0,
                 n_epochs = 1500,
                 verbose=True
                 ):

    	self.n_dims = n_dims
        self.n_neighbors = n_neighbors
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.lr = lr
        self.n_epochs = n_epochs
        self.verbose = verbose

        if self.n_dims < 2:
            raise ValueError('The number of output dimensions must be at least 2.')
        if self.n_neighbors < 1:
            raise ValueError('The number of nearest neighbors must be a positive number.')
        if self.n_outliers < 1:
            raise ValueError('The number of outliers must be a positive number.')
        if self.n_random < 0:
            raise ValueError('The number of random triplets must be a non-negative number.')
        if self.lr <= 0:
            raise ValueError('The learning rate must be a positive value.')


        if self.verbose:
            print("TRIMAP(n_neighbors={}, n_outliers={}, n_random={}, "
                  "lr={}, n_epochs={}, verbose={})".format(
                  n_neighbors, n_outliers, n_random, lr, n_epochs, verbose))

    def fit(self, X, init = None):
        """
        blah
        """
        X = X.astype(np.float64)
        
        self.embedding_ = trimap(X, self.n_dims, self.n_neighbors, self.n_outliers, self.n_random,
         self.lr, self.n_epochs, init, self.verbose)

        return self

    def fit_transform(self, X, init = None):
        """
        blah
        """
        self.fit(X, init)
        return self.embedding_

def load_known_size(filename):
    first = True
    with open(filename) as f:
        for irow, line in enumerate(f):
            if first:
                nrow, ncol = [int(a) for a in line.split(',')]
                x = np.empty((nrow, ncol), dtype = np.float64)
                first = False
            else:
                x[irow-1, :] = [float(a) for a in line.split(',')]
    return x

def main():
    filename = sys.argv[1]
    X = load_known_size(filename)
    Y = TRIMAP().fit_transform(X)
    np.savetxt('result.txt',Y)
    
if __name__ == '__main__':
    main()