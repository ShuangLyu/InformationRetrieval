import nmslib


class HNSW():
    def __init__(self, k_best, M, efC, num_threads, efS, space):
        self.K = k_best
        self.num_threads = num_threads
        self.index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC,
                                  'post': 0}
        self.index = nmslib.init(method='hnsw', space=space,
                                 data_type=nmslib.DataType.SPARSE_VECTOR)
        self.query_time_params = {'efSearch': efS}

    def createIdx(self, doc2vec):
        self.index.addDataPointBatch(doc2vec)
        self.index.createIndex(self.index_time_params, False)

    def querydoc(self, q):
        self.index.setQueryTimeParams(self.query_time_params)
        nbrs = self.index.knnQueryBatch(q, k=self.K, num_threads=self.num_threads)
        return nbrs
