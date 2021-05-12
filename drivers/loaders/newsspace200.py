import os
import wget
import bz2
import pandas as pd
import pandas as pd 
import xml.etree.ElementTree as et

class Newsspace200:
    def __init__(self, path, train_test_split=0.8):
        self.url = "http://groups.di.unipi.it/~gulli/newsspace200.xml.bz"
        self.dir = "/newsspace200"
        self.name = "Newsspace200"
        self.train_test_split = train_test_split
        if os.path.isdir(path + self.dir) == False:
            file = wget.download(self.url, out=path)
            print("DOWNLOADED:", self.url)

            os.mkdir(path + self.dir)

            zipfile = bz2.BZ2File(file)
            head, filename = os.path.split(file)
            data = zipfile.read()            
            with open(path + self.dir + "/" + filename[:-3], 'wb') as f:
                f.write(data)

            zipfile.close()
            
            if os.path.isfile(file):
                os.remove(file)
            print("EXTRACTED:", file)

            self.create_json(path + self.dir, filename[:-3])
            print("CREATED: train and test")

        self.test = pd.read_json(path + self.dir + "/test.json", orient="records", lines=True)
        self.train = pd.read_json(path + self.dir + "/train.json", orient="records", lines=True)
        
        self.max_labels = self.train.groupby(["label"]).count().index.max()

    def create_json(self, root, filename):
        xtree = et.parse(root + "/" + filename)
        xroot = xtree.getroot() 

        label = []
        text = []
        for node in xroot: 
            if node.tag=="category":
                label.append(node.text)
            elif node.tag=="description":
                text.append(node.text)
            else:
                continue

        df = pd.DataFrame()
        df["text"] = text
        df["label"] = label
        
        df = df[(df['label'].str.len()<15) & (df['label'].str.len()>1)]
        df = df.dropna()

        grouped_labels = df.groupby(["label"]).count().index
        grouped_labels = { grouped_labels[i] : i for i in range(len(grouped_labels))}
        df["label"] = df["label"].apply(lambda x : grouped_labels[x])

        labels = df.groupby(["label"]).count().index
        trains = pd.DataFrame()
        tests = pd.DataFrame()
        for label in labels:
            tmp = df[df["label"] == label]
            split = round(len(tmp) * self.train_test_split)
            trains = pd.concat([trains, tmp[:split]])
            tests = pd.concat([tests, tmp[split:]])

        trains = trains.sample(frac=1)
        trains.to_json(root + "/train.json", orient="records", lines=True)
        print("CREATED:", root + "/train.json")
        tests = trains.sample(frac=1)
        tests.to_json(root + "/test.json", orient="records", lines=True)
        print("CREATED:", root + "/test.json")

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
