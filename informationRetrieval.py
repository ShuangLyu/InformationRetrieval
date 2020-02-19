from getdoc2vec import GetDoc2vec
from randomcube import RandomCube
from hnsw_nmslib import HNSW
from KNNGraph import KNNG
from read_write import ReadWrite
import pandas as pd
import time
import numpy as np
import math
import scipy.sparse as ss
from apscheduler.schedulers.background import BackgroundScheduler


class InformationRetrieval():
    def __init__(self, stopwordspath, docspath, k_best, distance):
        self.readwrite = ReadWrite()
        self.k_best = k_best
        self.docs = self.readwrite.toloadcsv(file_path=docspath, usecols='TSNR')
        self.stopwords = self.readwrite.toloadtxt(file_path=stopwordspath)
        self.queryvec = GetDoc2vec(stopwords=self.stopwords)
        self.doc2vec = self.queryvec.tomatrix(self.docs)
        self.hnsw = HNSW(k_best=k_best, M=50, efC=200, num_threads=4, efS=200, space=distance)
        self.hnsw.createIdx(self.doc2vec)
        self.knng = KNNG(n_docs=self.doc2vec.shape[0], k_best=self.k_best, dist=distance)
        self.knng.buildgraph(self.doc2vec)

    def search(self, qdoc):
        self.doc = qdoc
        self.q2vec = self.queryvec.tomatrix(self.doc)
        try:
            t0 = time.time()
            start_docids = self.hnsw.querydoc(self.q2vec)[0][0]  # 生成搜索起点
            t1 = time.time()
            print("生成搜索起点用时：" + str(t1 - t0))
        except RuntimeError as r:
            print("很抱歉，输入的内容过于简单，请重新输入。")  # self.q2vec为零向量
            print(r)
        else:
            if len(start_docids) != 0:
                t2 = time.time()
                self.similarid = self.knng.search(self.q2vec, start_docids, self.doc2vec)
                t3 = time.time()
                print("search用时：" + str(t3 - t2))
                sim = self.q2vec.dot(self.doc2vec[self.similarid].T).toarray()[0]
                if sim[0] == 0:
                    print("很抱歉，没有找到相似内容，请检查您的输入是否正确。")
                else:
                    self.similarid = self.similarid[:len(np.nonzero(sim)[0])]
                    print(self.docs[self.docs.columns.values[0]][self.similarid])
                    print("检索完成，已为您找到%d条相似投诉" % (len(self.similarid)))
            else:
                print("很抱歉，没有找到相似内容，请检查您的输入是否正确。")

    def update(self):
        # 更新信访内容
        row = pd.DataFrame({self.docs.columns.values[0]: self.doc[0]}, index=[1])
        self.docs = self.docs.append(row, ignore_index=True)
        # 更新向量
        self.doc2vec = ss.vstack((self.doc2vec, self.q2vec))
        # 更新Graph
        self.knng.graph.append(self.similarid)
        for id in self.similarid:
            self.knng.graph[id].append(len(self.knng.graph) - 1)
            newgraphid = self.knng.graph[id]
            simindex = self.knng.sdist(self.doc2vec[id], self.doc2vec[newgraphid])[0]
            self.knng.graph[id] = [newgraphid[i] for i in simindex]

    def savealldata(self, docspath, doc2vecpath, graphpath):
        # 保存信访内容
        self.readwrite.tosavecsv(docspath, self.docs)
        # 保存向量
        self.readwrite.tosavepkl(doc2vecpath, self.doc2vec)
        # 保存Graph
        self.readwrite.tosavepkl(graphpath, self.knng.graph)
        # 更新并保存索引
        self.hnsw.createIdx(self.doc2vec)
        self.hnsw.index.saveIndex('sparse_index.bin', save_data=True)
        print('数据已保存')


if __name__ == "__main__":
    pd.set_option('max_colwidth', 100)
    ir = InformationRetrieval(stopwordspath='D:/xftest/0920/stopword.txt', docspath='D:/xftest/1106/tsnr.csv', \
                              k_best=40, distance='cosinesimil_sparse')
    # distance:‘cosinesimil_sparse’,‘l2_sparse’,‘l1_sparse’,‘linf_sparse’
    # 假设每天23点保存一次数据,并更新索引，线程在后台运行
    scheduler = BackgroundScheduler()
    scheduler.add_job(ir.savealldata, 'cron', hour='10', minute='51', args=['path', "path", "path"])
    scheduler.start()
    find = True
    while (find):
        qdoc = [(input("请输入查询:"))]
        ir.search(qdoc)
        update = input("是否更新(yes or no):")
        if update == "yes":
            ir.update()
