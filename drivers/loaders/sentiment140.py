import os
import wget
import zipfile
import pandas as pd

class Sentiment140:
    def __init__(self, path):
        self.url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        self.dir = "/trainingandtestdata"
        self.name = "Sentiment140"
        if os.path.isdir(path + self.dir) == False:
            file = wget.download(self.url, out=path)
            print("DOWNLOADED:", self.url)

            with zipfile.ZipFile(file, "r") as fzip:
                fzip.extractall(path + self.dir)

            if os.path.isfile(file):
                os.remove(file)
            print("EXTRACTED:", file)

        self.test = self.read(path + self.dir + "/testdata.manual.2009.06.14.csv")
        self.train = self.read(path + self.dir + "/training.1600000.processed.noemoticon.csv")

        self.max_labels = self.train.groupby(["label"]).count().index.max()

    def get_labels(self):
        return self.max_labels

    def get_test(self):
        return self.test

    def get_train(self):
        return self.train

    def read(self, filename):
        text = []
        label = []
        with open(filename, "r", encoding='cp1252') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.split(",")
                text.append(tmp[-1])
                label.append(int(tmp[0][1:2]))

        result = pd.DataFrame()
        result["text"] = text
        result["label"] = label

        return result
        