import numpy as np
import scipy.spatial.distance as sd
import matplotlib.pyplot as pyplot
from graph_helper import *
import sklearn.cluster as skc
from sklearn.neighbors import NearestNeighbors


def build_similarity(X,eps=None,k=0):

    
    if k > 0:
        knbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',metric='minkowski').fit(X)
        similarities = knbrs.kneighbors_graph(X).toarray()
        similarities = np.max(np.stack((similarities,similarities.T)),axis=0) # make it symetric
    else:
        var = 1
        dists = sd.cdist(X,X,'euclidean')
        similarities = np.exp(-dists/var)

        if eps == None: # we set eps as the minimum weight among non-zero weight of the spanning tree
            max_tree = max_span_tree(similarities)
            weights = max_tree*similarities
            weights = weights.flatten()
            weights[weights==0] = weights.max()
            eps = weights.min()

        similarities = similarities*(similarities>=eps)

    n = similarities.shape[0]
    similarities[np.arange(n),np.arange(n)] = 0
    return similarities

def build_laplacian(W, laplacian_normalization="rw"):
#  laplacian_normalization:
#      string selecting which version of the laplacian matrix to construct
#      either 'unn'normalized, 'sym'metric normalization
#      or 'rw' random-walk normalization

#################################################################
# build the laplacian                                           #
# L: (n x n) dimensional matrix representing                    #
#    the Laplacian of the graph                                 #
#################################################################
    n = W.shape[0]
    D = np.zeros((n,n))
    diag = np.zeros(n)
    diag[:] = W.sum(axis=0)
    D[range(n),range(n)] = diag[:]
    L = np.zeros((n,n))
    L = D - W
    if laplacian_normalization == 'unn':
        return L
    elif laplacian_normalization == 'sym':
        assert (diag>0).all(), ("at least 1 node has degree = 0", diag)
        D_halfinv = np.zeros((n,n))
        D_halfinv[range(n),range(n)] = diag**(-1/2)
        return D_halfinv.dot(L).dot(D_halfinv)
    elif laplacian_normalization == 'rw':
        assert (diag>0).all(), "at least 1 node has degree = 0"
        D_inv = np.zeros((n,n))
        D_inv[range(n),range(n)] = diag**(-1)
        return D_inv.dot(L)
    else:
        assert False, "unexpected option in laplacian_normalization"


def spectral_analysis(L,k,chosen_eig_indices=None,do_clusters=False,do_plots=False,num_classes=2,EPSILON_IS_0=10**(-7)):
    assert do_plots or do_clusters, "Task must be either to plot or to assign clusters"

    # U = (n x n) eigenvector matrix
    # E = (n x n) eigenvalue diagonal matrix (sorted)

    n = L.shape[0]
    if chosen_eig_indices == None:
        chosen_eig_indices = np.arange(k)

    S,U = scipy.sparse.linalg.eigs(L,k=k) # eigen elements are sorted in descending order
    ordre = np.argsort(np.abs(S)) # ascending order
    S = S[ordre] # ie reordering eigen elements in ascending order
    print(S)
    U = U[:,ordre]

    U_chosen = U[:,chosen_eig_indices]
    U_chosen[np.abs(U_chosen)<EPSILON_IS_0] = 0
    print((U_chosen!=0).sum(axis=0),U_chosen.shape)
    rows_norm = np.linalg.norm(U_chosen, axis=1, ord=2)
    U_chosen[rows_norm<=0,:] = 0
    U_chosen[rows_norm>0,:] = (U_chosen[rows_norm>0,:].T / rows_norm[rows_norm>0]).T

    if do_plots:
        # insightful plots of the eigen vecors
        pyplot.figure(figsize=(20,10))
        pyplot.subplot(2,6,1)
        pyplot.title("Eigen values")
        pyplot.plot(S)
        for i in range(len(chosen_eig_indices)):
            pyplot.subplot(2,6,2+i)
            eig = np.sort(U_chosen[:,i])
            pyplot.title("Eig vect # %d"%i)
            pyplot.plot(eig)

    if do_clusters:
        Y = np.zeros(n)
        assignment_method = "kmean"
        if assignment_method == "thresh":
            for i in range(len(chosen_eig_indices)):
                eig = U_chosen[:,i]
                Y[np.abs(eig)<EPSILON_IS_0] += 3**i # encoding in base 3, separate zero and non-zeros
                Y[eig>=EPSILON_IS_0] += 2*3**i # encoding in base 3, separate positive and negative
            uniques = np.unique(Y)
            print("[Threshold method] Unique assignement codes are",uniques)
            if len(uniques) > 8:
                print("[Threshold method] Plot functions limit the number of colors to 6 ... some clusters need to be merged")
            for i in range(len(uniques)):
                Y[Y==uniques[i]] = i%8 # mod 8 : this is for compatibility with helper.plot_edges_and_points that can plot only 8 colors
        elif assignment_method == "kmean":
            Y[:] = skc.KMeans(num_classes).fit_predict(np.abs(U_chosen))
        else:
            assert False, "assignment_method must be either 'thresh' or 'kmean'"

        return Y




