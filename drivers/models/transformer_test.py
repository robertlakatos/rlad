import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformer import Transformer

MAX_LEN=128
VOCAB_SIZE=1000

train = pd.read_json("./encodes/word_level_IMDB_train.json", 
                     orient="records", 
                     lines=True)

test = pd.read_json("./encodes/word_level_IMDB_test.json", 
                     orient="records", 
                     lines=True)

transformer = Transformer(vocab_size=VOCAB_SIZE, 
                          input_length=MAX_LEN,
                          output_size=1,
                          repeate=1,
                          N=2,
                          embedding_size=8,
                          train_len=len(train["X"].values))

# transformer.create_model()
# transformer.summary()

history = transformer.fit(train_X=pad_sequences(train["X"].values, maxlen=MAX_LEN), 
                          train_y=np.array(train["y"].values), 
                          val_X=pad_sequences(test["X"].values, maxlen=MAX_LEN), 
                          val_y=np.array(test["y"].values),
                          summary=True)