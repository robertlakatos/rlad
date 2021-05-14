import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from drivers.loaders.imdb import IMDB
from drivers.loaders.newsspace200 import Newsspace200
from drivers.loaders.sentimentMD import SentimentMD
from drivers.loaders.sentiment140 import Sentiment140

from drivers.tokenizers.word_piece_vocab import WordPieceVocab
from drivers.tokenizers.word_level_vocab import WordLevelVocab
from drivers.tokenizers.unigram_vocab import UnigramVocab
from drivers.tokenizers.bpe_vocab import BPEVocab

from drivers.models.simple import Simple

class EvalT:
    def __init__(self, vocab_size=1000, max_pedding_ratio=0.8, repeate=2, tokenizer_vocab=None):
        self.name = "EvalT"
        self.vocab_size = vocab_size
        self.unk_token = "[UNK]"
        self.path_data = "data"
        self.path_vocabs = "vocabs/"
        self.path_encodes = "encodes/"
        self.max_pedding_ratio = max_pedding_ratio
        self.repeate = repeate
        self.tokenizer_vocab = tokenizer_vocab
        self.history = []

        if os.path.isdir(self.path_data) == False:
            os.mkdir(self.path_data)

        if os.path.isdir(self.path_vocabs) == False:
            os.mkdir(self.path_vocabs)

        if os.path.isdir(self.path_encodes) == False:
            os.mkdir(self.path_encodes)

        self.dbs = [
            { "db" : SentimentMD(self.path_data) },
            { "db" : Sentiment140(self.path_data) },
            { "db" : Newsspace200(self.path_data) },
            { "db" : IMDB(self.path_data) }
        ]
       
        self._setup_tokenizers()

    def run(self):
        for item in self.dbs:
            for i in range(len(item["vocabs"])):
                model_name = "Simple"
                simple_name = model_name + "_" + item["db"].name + "_" + item["vocabs"][i].name

                simple = Simple(vocab_size=item["vocabs"][i].get_vocab_size(), 
                                input_lenght=len(item["encodes"][i]["train"]["X"].values[0]), 
                                embedding_size=8,
                                output_size=item["db"].get_labels(),
                                repeate=self.repeate,
                                name=simple_name)
                
                simple.set_data(train_X=np.array([item for item in item["encodes"][i]["train"]["X"].values]), 
                                train_y=np.array([item for item in item["encodes"][i]["train"]["y"].values]), 
                                test_X=np.array([item for item in item["encodes"][i]["test"]["X"].values]), 
                                test_y=np.array([item for item in item["encodes"][i]["test"]["y"].values]))

                history = simple.fit()
                
                print(model_name, item["db"].name, item["vocabs"][i].name)
                
                for h in history: print(h.history)

                self.history.append({
                    "model" : model_name,
                    "db" : item["db"].name,
                    "vocab" : item["vocabs"][i].name,
                    "history" : history
                })

    def _setup_tokenizers(self):
        for item in self.dbs:
            item["vocabs"] = [
                WordPieceVocab(item["db"].get_train()["text"].values, self.unk_token, self.vocab_size),
                WordLevelVocab(item["db"].get_train()["text"].values, self.unk_token, self.vocab_size),
                UnigramVocab(item["db"].get_train()["text"].values, self.unk_token, self.vocab_size),
                BPEVocab(item["db"].get_train()["text"].values, self.unk_token, self.vocab_size)
            ]

            if self.tokenizer_vocab is not None:
                item["vocabs"].append(self.tokenizer_vocab)

        for item in self.dbs:
            for vocab in item["vocabs"]:
                file_name_vocabs = self.path_vocabs + vocab.name + "_" + item["db"].name + ".json"
                print(file_name_vocabs)
                if os.path.isfile(file_name_vocabs) == False:
                    vocab.train()
                    print("TRAINED:", item["db"].name)
                    vocab.save(file_name_vocabs)
                    print("SAVED:", item["db"].name)
                else:
                    vocab.load(file_name_vocabs)
                    print("LOADED:", item["db"].name)

        for item in self.dbs:
            item["encodes"] = []
            for vocab in item["vocabs"]:
                file_name_encodes_train = self.path_encodes + vocab.name + "_" + item["db"].name + "_train.json"
                file_name_encodes_test = self.path_encodes + vocab.name + "_" + item["db"].name + "_test.json"
                
                item["encodes"].append({
                    "train" : pd.DataFrame(),
                    "test" : pd.DataFrame(),
                })

                if os.path.isfile(file_name_encodes_train) == False or os.path.isfile(file_name_encodes_test) == False:
                    item["encodes"][-1]["train"]["X"] = self._encode(vocab, item["db"].get_train()["text"].values)
                    item["encodes"][-1]["test"]["X"] = self._encode(vocab, item["db"].get_test()["text"].values)
                                
                    if item["db"].get_labels() > 1:
                        train_y = tf.one_hot(item["db"].get_train()["label"].values, item["db"].get_labels()+1).numpy()
                        train_y = [list(item) for item in train_y]         
                        item["encodes"][-1]["train"]["y"] = train_y

                        test_y = tf.one_hot(item["db"].get_test()["label"].values, item["db"].get_labels()+1).numpy()      
                        test_y = [list(item) for item in test_y]
                        item["encodes"][-1]["test"]["y"] = test_y
                    else:
                        item["encodes"][-1]["train"]["y"] = item["db"].get_train()["label"].values                        
                        item["encodes"][-1]["test"]["y"] = item["db"].get_test()["label"].values
                    
                    item["encodes"][-1]["train"].to_json(file_name_encodes_train, orient="records", lines=True)
                    item["encodes"][-1]["test"].to_json(file_name_encodes_test, orient="records", lines=True)

                    print("ENCODED (CREATED AND LOADED):", 
                          file_name_encodes_train, 
                          file_name_encodes_test,
                          vocab.name, 
                          item["db"].name)
                else:                        
                    item["encodes"][-1]["train"] = pd.read_json(file_name_encodes_train, orient="records", lines=True)
                    item["encodes"][-1]["test"] = pd.read_json(file_name_encodes_test, orient="records", lines=True)
                    print("ENCODED (LOADED):", 
                        file_name_encodes_train, 
                        file_name_encodes_test, 
                        vocab.name, 
                        item["db"].name)

                tmp_sorted = list(item["encodes"][-1]["train"].X.map(len).sort_values())
                index = round(len(tmp_sorted) * self.max_pedding_ratio)
                item["encodes"][-1]["train"]["X"] = list(pad_sequences(item["encodes"][-1]["train"]["X"].values, 
                                                                    maxlen=tmp_sorted[index]))
                item["encodes"][-1]["test"]["X"] = list(pad_sequences(item["encodes"][-1]["test"]["X"].values, 
                                                                    maxlen=tmp_sorted[index]))
                print("PADDED TRAIN AND TEST: ", tmp_sorted[index], vocab.name, item["db"].name)    

    def _encode(self, model, data):
        result = []
        for item in data:
            output = model.encode(item)
            result.append(output.ids)
        return result