import os
import json
import wget
import tarfile
import pandas as pd

class IMDB:
  def __init__(self, path):
      self.url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
      self.dir = "/aclImdb"
      self.name = "IMDB"
      if os.path.isdir(path + self.dir) == False:
          file = wget.download(self.url, out=path)
          print("DOWNLOADED:", self.url)

          tar = tarfile.open(file, "r:gz")
          tar.extractall(path)
          tar.close()

          if os.path.isfile(file):
              os.remove(file)
          print("EXTRACTED:", file)

      if os.path.exists(path + self.dir + "/test.json") == False:
          self.create_json(path, self.dir + "/test")

      if os.path.exists(path + "/aclImdb/train.json") == False:
          self.create_json(path, self.dir + "/train")
    
      self.test = pd.read_json(path + self.dir + "/test.json", orient="records", lines=True)
      self.train = pd.read_json(path + self.dir + "/train.json", orient="records", lines=True)

      self.max_labels = self.train.groupby(["label"]).count().index.max()

  def get_labels(self):
      return self.max_labels

  def get_test(self):
      return self.test

  def get_train(self):
      return self.train

  def create_json(self, path, mark):
      test_dir_neg = path + "/" + mark +"/neg/"
      test_dir_pos = path + "/" + mark +"/pos/"
      
      buff = pd.concat([self.add_label(test_dir_neg, 0),
                        self.add_label(test_dir_pos, 1)])
      buff = buff.sample(frac=1)
      
      buff.to_json(path + "/" + mark + ".json",orient="records", lines=True)

  def add_label(self, directory, label):     
      files = os.listdir(directory)
      text = []
      for file in files:
          text.append(self.read(directory + file))
          print("IMDB READED:", file)

      result = pd.DataFrame()
      result["text"] = text
      result["label"] = [label] * len(text)
      return result
    

  def read(self, filename):
     with open(filename, "r", encoding='utf-8') as f:
         return f.read()
