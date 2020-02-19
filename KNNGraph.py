import numpy as np
import pandas as pd
import read_write
import time
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.spatial.distance import chebyshev
import pairwise_fast



# 构建图查找出kbest个相似文档
class KNNG():

    def __init__(self, n_docs, k_best, dist):
        self.graph = []
        self.n_docs = n_docs
        self.k_best = k_best
        self.similar = []
        self.y_norm_squared = np.random.randint(1,2,n_docs,dtype='int32').reshape(1,n_docs)
        self.dist = dist

    def sdist(self, v1, v2):
        distance = []
        if self.dist == 'cosinesimil_sparse':
            for i in range(v1.shape[0]):
                s = 1 - v1[i].dot(v2.T).toarray()
                distance.append(np.argsort(s).tolist()[0][0:self.k_best])
            return distance
        elif self.dist == 'l1_sparse':
            for i in range(v1.shape[0]):
                D = np.zeros((v1[i].shape[0], v2.shape[0]))
                s = pairwise_fast._sparse_manhattan(v1[i].data, v1[i].indices, v1[i].indptr, \
                                                    v2.data, v2.indices, v2.indptr, D)
                distance.append(np.argsort(s[0])[0:self.k_best].tolist())
            return distance
        elif self.dist == 'l2_sparse':
            for i in range(v1.shape[0]):
                s = euclidean_distances(v1[i], v2)  #,Y_norm_squared=self.y_norm_squared[:,:v2.shape[0]]
                distance.append(np.argsort(s).tolist()[0][0:self.k_best])
            return distance
        elif self.dist == 'linf_sparse':
            for i in range(v1.shape[0]):
                D = np.zeros((v1[i].shape[0], v2.shape[0]))
                s = pairwise_fast._sparse_chebyshev(v1[i].data, v1[i].indices, v1[i].indptr, \
                                                    v2.data, v2.indices, v2.indptr, D)
                distance.append(np.argsort(s[0])[0:self.k_best].tolist())
            return distance

    def buildgraph(self, docvec):
        self.graph = self.sdist(docvec, docvec)
        print(self.graph)

    def search(self, q, start_docsid, doc2vec):
        startdoc = set()
        for i in start_docsid:
            startdoc.update(self.graph[i])
        startdoc = list(startdoc)
        start_docs = doc2vec[startdoc]
        simindex = self.sdist(q, start_docs)[0]
        result = []
        for index in simindex:
            result.append(startdoc[index])
        return result

