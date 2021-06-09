# https://arxiv.org/pdf/1803.01271.pdf
# https://www.programmersought.com/article/13674618779/
# https://www.tensorflow.org/addons/tutorials/layers_weightnormalization
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense

class ACBiLSTM:
    def __init__(self, vocab_size=None, input_length=None, embedding_size=300, dropout=0.0,
                 output_size=1, patience=2, batch_size=50, repeate=10, epochs=1000):
        super().__init__()

        self.name = "ACBiLSTM"
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.output_size = output_size
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.repeate = repeate        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.model = None
        self.history = []
        
        self.metrics = ["accuracy"]
        self.metrics_multi_label = False
        self.metrics_specificity = 0.5
        self.metrics_sensitivity = 0.5

        self.activation = "sigmoid"
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.monitor = "val_loss" # "val_binary_accuracy"

        if self.output_size > 1:
            self.activation = "softmax"

        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor=self.monitor, 
                                                               patience=self.patience)
            

    def create_model(self):
        
        inputs = Input(shape=(self.input_length,))

        emb = Embedding(input_dim=self.vocab_size,
                        output_dim=self.embedding_size,
                        input_length=self.input_length)(inputs)

        conv = Conv1D(filters=100,
                      kernel_size=3,
                      strides=1,
                      activation="relu")(emb)

        lstm_forward = LSTM(units=150,
                            activation="tanh",
                            name="LSTM_forward",
                            return_sequences=True)(conv)
        lstm_backward = LSTM(units=150,
                             activation="tanh",
                             name="LSTM_backward",
                             return_sequences=True,
                             go_backwards=True)(conv)

        attention_forward = Attention(name="attention_forward")([lstm_forward, lstm_forward])
        attention_backward = Attention(name="attention_backward")([lstm_backward, lstm_backward])
        
        flatten_forward = Flatten(name="flatten_forward")(attention_forward)
        flatten_backward = Flatten(name="flatten_backward")(attention_backward)

        comprehensive_context = Concatenate()([flatten_forward, flatten_backward])
        
        dropout = Dropout(self.dropout)(comprehensive_context)
                
        outputs = Dense(units=self.output_size, activation=self.activation)(dropout)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)

        self.model.compile(optimizer='adam',
                           loss=self.loss,
                           metrics=self.metrics)

    def summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print("Model is not initialized")

    def _set_data(self, train_X, train_y, val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y

    def get_history(self):
        return self.history

    def fit(self, train_X, train_y, val_X, val_y, summary=False):
        self.create_model()

        if summary == True:
            self.summary()

        self._set_data(train_X, train_y, val_X, val_y)

        self.history = self.model.fit(train_X, 
                                      train_y, 
                                      epochs=self.epochs, 
                                      callbacks=self.early_stopping, 
                                      batch_size=self.batch_size, 
                                      validation_data=(val_X, val_y))

        return self.history

    def __call__(self, inputs):
        return self.model(inputs)