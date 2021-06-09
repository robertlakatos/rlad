# https://arxiv.org/pdf/1803.01271.pdf
# https://www.programmersought.com/article/13674618779/
# https://www.tensorflow.org/addons/tutorials/layers_weightnormalization
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout

class ResidualBlock(Layer):
    def __init__(self, filters=32, kernel_size=3, dilation_rate=1, dropout=[0.0,0.0], name=None):
        super(ResidualBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size 
        self.dilation_rate = dilation_rate
        self.dropout = dropout

        self.conv1 = tfa.layers.WeightNormalization(Conv1D(filters=filters, 
                                                           kernel_size=kernel_size,
                                                           padding='same',
                                                           dilation_rate=dilation_rate))
        self.conv2 = tfa.layers.WeightNormalization(Conv1D(filters=filters,
                                                           kernel_size=kernel_size,
                                                           padding='same',
                                                           dilation_rate=dilation_rate))

        self.act1 = Activation('relu')
        self.act2 = Activation('relu')

        self.dropout1 = Dropout(self.dropout[0])
        self.dropout2 = Dropout(self.dropout[1])

    def call(self, x): 
        out = self.conv1(x)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.conv2(x)
        out = self.act2(out)
        out = self.dropout2(out)

        return out

class TCNN:
    def __init__(self, vocab_size=None, input_length=None, embedding_size=300,
                 output_size=1, patience=2, batch_size=50, repeate=10, epochs=1000):
        super().__init__()

        self.name = "TCNN"
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.embedding_size = embedding_size
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
        self.model = tf.keras.Sequential(name=self.name)
         
        self.model.add(tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                 output_dim=self.embedding_size,
                                                 input_length=self.input_length))

        self.model.add(ResidualBlock(filters=32, kernel_size=3, dilation_rate=1))
        self.model.add(ResidualBlock(filters=32, kernel_size=3, dilation_rate=2))
        self.model.add(ResidualBlock(filters=16,kernel_size=3,dilation_rate=4))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(units=self.output_size,
                                             activation=self.activation))

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