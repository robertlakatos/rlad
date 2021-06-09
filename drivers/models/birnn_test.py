import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from birnn import BiRNN

MAX_LEN=128
VOCAB_SIZE=1000

train = pd.read_json("./encodes/word_level_IMDB_train.json", 
                     orient="records", 
                     lines=True)

test = pd.read_json("./encodes/word_level_IMDB_test.json", 
                     orient="records", 
                     lines=True)

lstm = BiRNN(vocab_size=VOCAB_SIZE,
             input_length=MAX_LEN, 
             embedding_size=8,
             output_size=1,
             repeate=1,
             name="lstm",
             layers=[{"activation":"tanh", "dropout" : 0.5},
                     {"activation":"tanh", "dropout" : 0.0}])

history = lstm.fit(train_X=pad_sequences(train["X"].values, maxlen=MAX_LEN), 
                   train_y=np.array(train["y"].values), 
                   val_X=pad_sequences(test["X"].values, maxlen=MAX_LEN), 
                   val_y=np.array(test["y"].values),
                   summary=True)