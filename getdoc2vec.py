from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import  numpy as np
import pandas as pd
import codecs
from sklearn import preprocessing
from read_write import ReadWrite

#类GetDoc2vec()形成词向量矩阵
class GetDoc2vec():

    def __init__(self,stopwords):
        self.stopwords = stopwords
        self.stopwordlist = [w.strip() for w in self.stopwords]
        self.tfidfmodel = TfidfVectorizer()
        self.rw = ReadWrite()

    def tokenization(self, doc):
        words = jieba.cut(doc)
        result = " ".join(words)
        return result

    def tomatrix(self, docs):
        dockeys = []
        if len(docs) > 1:
            for row in docs.itertuples():
                print(row.Index)
                soup = getattr(row, docs.columns[0])
                word_list = self.tokenization(soup)
                dockeys.append(word_list)
            self.tfidfmodel = TfidfVectorizer().fit(dockeys) #stop_words = self.stopwordlist
            sparse_result = self.tfidfmodel.transform(dockeys)
            return sparse_result
        else:   #将一条文档向量化
            word_list = self.tokenization(docs[0])
            dockeys.append(word_list)
            sparse_result = self.tfidfmodel.transform(dockeys)
            return sparse_result
