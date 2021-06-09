import os
import sys
import inspect
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,root) 

from drivers.models.cnn import CNN

MAX_LEN=128
VOCAB_SIZE=1000

train = pd.read_json("./encodes/word_level_IMDB_train.json", 
                     orient="records", 
                     lines=True)

test = pd.read_json("./encodes/word_level_IMDB_test.json", 
                     orient="records", 
                     lines=True)

cnn = CNN(vocab_size=VOCAB_SIZE, 
                input_length=MAX_LEN, 
                embedding_size=8,
                output_size=1,
                repeate=1)

history = cnn.fit(train_X=pad_sequences(train["X"].values, maxlen=MAX_LEN), 
                     train_y=np.array(train["y"].values), 
                     val_X=pad_sequences(test["X"].values, maxlen=MAX_LEN), 
                     val_y=np.array(test["y"].values),
                     summary=True)