import matplotlib.pyplot as pyplot
import scipy.misc as sm
import numpy as np
import cv2 as cv
import os
import sys
from scipy.spatial import distance
import scipy.io as sio
from tqdm import tqdm

path=os.path.dirname(os.getcwd())
sys.path.append(path)
from graph_helper import *



class incremental_k_centers:  
    def __init__(self, labeled_samples, labels, max_num_centroids = 50):
        ## Number of labels to cluster
        self.n_labels = max(labels)
        ## Dimension of the input image
        self.sample_dimension = labeled_samples.shape[1]
        ## Check input validity
        assert (set(labels) == set(range(1, 1 + self.n_labels))), "Initially provided samples should be labeled in [1, max]"
        assert (len(labeled_samples) == len(labels)), "Initial samples and initial labels are not of same size"
        ## Number of labeled samples
        self.n_labeled_samples = len(labeled_samples)
        ## Model parameter : number of maximum stored centroids
        self.max_num_centroids = max_num_centroids
        ## Model centroids (inital labeled samples)
        self.centroids = labeled_samples
        ## Centroids labels
        self.Y = labels
        ## Compute all the distances
        self.centroids_distances = None
        self.init = True
        #self.taboo = (np.zeros(self.n_labeled_samples) == 0)
        
    def online_ssl_update_centroids(self, sample):
    #
    # Input

    # self:
    #     the current cover state
    # sample:
    #     the new sample
    # Output
    # No output, update self   
        assert (self.sample_dimension == len(sample)), "new sample not of good size {0}".format(self.sample_dimension)
        
        if self.centroids.shape[0] >= self.max_num_centroids + 1:
            
            # Initialization
            if self.init:
                ## Compute the centroids distances 
                self.centroids_distances = distance.cdist(self.centroids, self.centroids)
                ## set labeled nodes and self loops as infinitely distant
                np.fill_diagonal(self.centroids_distances, +np.Inf)
                self.centroids_distances[0:self.n_labeled_samples, 0:self.n_labeled_samples] = +np.Inf
                ## put labeled nodes in the taboo list
                self.taboo = np.array(range(self.centroids.shape[0])) < self.n_labeled_samples
                ## initialize multiplicity
                self.V = np.ones(self.centroids.shape[0])
                self.init = False
            # find the edge (c_rep,c_add) with the minimum distance 
            min_dist = self.centroids_distances.min()
            c_rep, c_add = np.unravel_index(self.centroids_distances.argmin(), self.centroids_distances.shape)

            # update data structures
            # None of the two nodes are taboo, if there is one, it is c_rep,
            # otherwise, c_rep is the bigest centroid
            if (c_rep in self.taboo) and (c_add in self.taboo):
                assert False, "Algorithm assumption: 2 closest centroids cannot be both in the taboo list"
            elif (c_rep in self.taboo):
                pass
            elif (c_add in self.taboo):
                c_rep, c_add = c_add, c_rep
            else:
                if self.V[c_rep] < self.V[c_add]:
                    c_rep, c_add = c_add, c_rep
            
            # c_rep absorbe c_add, c_add is now the new sample
            self.V[c_rep] += self.V[c_add]
            self.V[c_add] = 1
            self.centroids[c_add] = sample   
   
            ## Update the matrix distance
            dist_row = distance.cdist([self.centroids[c_add]], self.centroids)[0]
            dist_row[c_add] = +np.inf
            self.centroids_distances[c_add, :] = dist_row
            self.centroids_distances[:, c_add] = dist_row
            self.last_sample = c_add
            
        else:
            ## Just add the new sample as a centroid
            current_len = len(self.centroids)
            self.Y = np.append(self.Y, 0)
            self.centroids = np.vstack( [self.centroids, sample] )
            self.last_sample = current_len-1 # modif
            
            
    def online_ssl_compute_solution(self, var, eps, k):
    # Output
    # A label vector prediction of self.last_sample

        W = build_similarity_graph(self.centroids, var = var, eps = eps, k = k)
        if self.init:
            V=np.diag(np.ones(self.centroids.shape[0]))
            self.last_sample = self.centroids.shape[0] - 1
        else:
            V=np.diag(self.V)
        W=V.dot(W.dot(V))
         
        L = build_laplacian(W, laplacian_normalization="", laplacian_regularization=0.2)

        f = hardHFS(graph=W, labels=self.Y, laplacian=L)

        # print(f)

        return f[self.last_sample]


def iterative_hfs(samples, idx_lbl, labels_binary, var, eps, k, niter = 20, laplacian_regularization=0):
    # load the data   
    # a skeleton function to perform HFS, needs to be completed
    #  Input
    #  samples:
    #      n samples of dimension d, shape (n,d)
    #  idx_lbl:
    #      list of indexes of samples of known labels
    #  labels_binary:
    #      one-hot array of shape (len(idx_lbl),K) where K is the number of categories
    #  var:
    #      for similarity funciton, normalization on the eucliean distances before applying exp
    #  eps:
    #      if eps != 0, build an eps-graph
    #  k:
    #      if k != 0, build a knn-graph
    #  niter:
    #      number of iterations to use for the iterative propagation
    #  laplacian_regularization:
    #      lambda in the Laplacian regularization (ie the 'sink' probabilty in random walk)


    #  Output
    #  labels:
    #      class assignments for each (n) nodes

    # Y_masked = np.zeros((samples.shape[0],1))
    # Y_masked[idx_lbl,0] = labels
    # classes = np.unique(Y_masked[Y_masked > 0])

    # # Compute the initializion vector f
    # f = (Y_masked == classes).astype(np.float)
    # # assert f.shape == (len(Y_masked),len(classes))

    K = labels_binary.shape[1]
    n = samples.shape[0]

    classes = np.arange(1,K+1,1)

    f = np.zeros((n,K))
    f[idx_lbl,:] = labels_binary

    W = build_similarity_graph(samples, var = var, eps = eps, k = k)
    


    # proceed to iterated averaging
    sum_w = W.sum(axis=0) + laplacian_regularization
    for it in range(niter):
        for cl_ind in range(len(classes)):
            f[:,cl_ind] = (W.dot(f[:,cl_ind]))/sum_w
    
    # Assign the label in {1,...,c}
    labels = classes[f.argmax(axis=1)]

    return labels,f
    


