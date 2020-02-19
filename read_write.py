import pickle
import pandas as pd


class ReadWrite():
    def tosavepkl(self, file_path, data):
        pickle.dump(data, open(file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def toloadpkl(self, file_path):
        return pickle.load(open(file_path, 'rb'))

    def tosavecsv(self, file_path, data, encoding="ansi"):
        data.to_csv(file_path, ',', encoding=encoding)

    def toloadcsv(self, file_path, usecols, encoding="ansi"):
        return pd.read_csv(file_path, ',', encoding=encoding, header=0, usecols=[usecols]).astype(str)

    def toloadtxt(self, file_path):
        f = open(file_path, "r", encoding='utf-8')
        stopwords = f.read().split("\n")
        return stopwords
