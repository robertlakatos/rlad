import os
import json
import wget
import tarfile
import pandas as pd
from lxml import etree

class SentimentMD:
    def __init__(self, path):
        self.url = "http://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz"
        self.dir = "/sorted_data/"
        self.name = "SentimentMD"
        if os.path.isdir(path + self.dir) == False:
            file = wget.download(self.url, out=path)
            print("DOWNLOADED:", self.url)

            tar = tarfile.open(file, "r:gz")
            tar.extractall(path)
            tar.close()

            if os.path.isfile(file):
                os.remove(file)
            print("EXTRACTED:", file)

            self.create_json(path + self.dir)
            print("CREATED: train and test")

        self.test = pd.read_json(path + self.dir + "test.json", orient="records", lines=True)
        self.train = pd.read_json(path + self.dir + "train.json", orient="records", lines=True)

        self.max_labels = self.train.groupby(["label"]).count().index.max()
    
    def get_labels(self):
        return self.max_labels
        
    def get_test(self):
        return self.test

    def get_train(self):
        return self.train

    def create_json(self, path):
        items = os.listdir(path)
        items = [path + item for item in items]
        files = [item + "/all.review" for item in items if os.path.isdir(item)]

        parser = etree.XMLParser(recover=True)
        trains = pd.DataFrame()
        tests = pd.DataFrame()

        for file in files:
            with open(file, "r", encoding="latin-1") as f:
                label = []
                text = []
                tree = etree.fromstring("<root>" + f.read() + "</root>", parser=parser)
                for elements in tree:
                    found_review_text = False
                    found_rating = False
                    for element in elements:
                        if element.tag == "review_text":
                            found_review_text = True
                            review_text = element.text
                            
                        if element.tag == "rating":
                            found_rating = True
                            rating = float(element.text)

                    if found_review_text and found_rating:
                        text.append(review_text)
                        label.append(rating)

                (sub_train, sub_test) = self.split_test_train(text, label)
                trains = pd.concat([trains, sub_train])
                tests = pd.concat([tests, sub_test])

            print("READED:", file)

        trains = trains.sample(frac=1)
        trains.to_json(path + "/train.json", orient="records", lines=True)
        tests = tests.sample(frac=1)
        tests.to_json(path + "/test.json", orient="records", lines=True)

    def split_test_train(self, text, label):
        df = pd.DataFrame()
        df["text"] = text
        df["label"] = label

        labels = df.groupby(["label"]).count().index
        trains = pd.DataFrame()
        tests = pd.DataFrame()
        for label in labels:
            tmp = df[df["label"] == label]
            split = round(len(tmp) * 0.8)
            trains = pd.concat([trains, tmp[:split]])
            tests = pd.concat([tests, tmp[split:]])

        return (trains, tests)
