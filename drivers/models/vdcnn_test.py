import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from vdcnn import VDCNN

MAX_LEN=128
VOCAB_SIZE=1000

train = pd.read_json("./encodes/word_level_IMDB_train.json", 
                     orient="records", 
                     lines=True)

test = pd.read_json("./encodes/word_level_IMDB_test.json", 
                     orient="records", 
                     lines=True)

# Alternatives call
# depth=17, shortcut=True, pool_type='k_max', proj_type='identity'
# depth=29, shortcut=False, pool_type='max', proj_type='conv'
# depth=49, shortcut=True, pool_type='conv', proj_type='conv'
vdcnn = VDCNN(vocab_size=VOCAB_SIZE, 
              input_length=MAX_LEN,
              depth=9, 
              shortcut=True, 
              pool_type='k_max', 
              proj_type='identity',
              embedding_size=16,
              batch_size=50,
              output_size=1,
              repeate=1)

history = vdcnn.fit(train_X=pad_sequences(train["X"].values, maxlen=MAX_LEN), 
                    train_y=np.array(train["y"].values), 
                    val_X=pad_sequences(test["X"].values, maxlen=MAX_LEN), 
                    val_y=np.array(test["y"].values),
                    summary=True)