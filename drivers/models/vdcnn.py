# https://arxiv.org/pdf/1606.01781.pdf
# https://github.com/cjiang2/VDCNN/blob/main/vdcnn.py

import tensorflow as tf
from tensorflow.keras import Model, layers

N_BLOCKS = {9: (1, 1, 1, 1), 
            17: (2, 2, 2, 2),
            29: (5, 5, 2, 2),
            49:(8, 8, 5, 3)}

class KMaxPooling(layers.Layer):
    def __init__(self, 
                 k=None, 
                 sorted=False):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.sorted = sorted

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k, input_shape[2])

    def call(self, inputs):
        if self.k is None:
            k = int(tf.round(inputs.shape[1] / 2))
        else:
            k = self.k

        shifted_inputs = tf.transpose(inputs, [0, 2, 1])
        top_k = tf.nn.top_k(shifted_inputs, k=k, sorted=self.sorted)[0]

        return tf.transpose(top_k, [0, 2, 1])

class Pooling(layers.Layer):
    def __init__(self, pool_type='max', name=None):
        super(Pooling, self).__init__(name=name)
        assert pool_type in ['max', 'k_max']
        self.pool_type = pool_type

        if pool_type == 'max':
            self.pool = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')
        elif pool_type == 'k_max':
            self.pool = KMaxPooling()
        
    def call(self, x):
        return self.pool(x)

class ZeroPadding(layers.Layer):
    def __init__(self, values, name=None):
        super(ZeroPadding, self).__init__(name=name)
        self.values = values

    def call(self, x):
        x = tf.pad(x, [[0, 0], [0, 0], [self.values[0], self.values[1]]], 
                   mode='CONSTANT', constant_values=0)
        return x

class Conv1D_BN(layers.Layer):
    def __init__(self, 
                 filters,
                 kernel_size=3,
                 strides=2,
                 padding='same',
                 use_bias=True,
                 name=None):
        super(Conv1D_BN, self).__init__(name=name)
        self.filters = filters
        self.use_bias = use_bias
        self.conv = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                                  kernel_initializer='he_normal')
        self.bn = layers.BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, use_bias=True, shortcut=True,
                 pool_type=None, proj_type=None, name=None):
        super(ConvBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.shortcut = shortcut
        self.pool_type = pool_type
        self.proj_type = proj_type

        # Deal with downsample and pooling
        assert pool_type in ['max', 'k_max', 'conv', None]
        
        if pool_type is None:
            strides = 1
            self.pool = None
            self.downsample = None
        elif pool_type == 'conv':
            strides = 2     # Convolutional pooling with stride 2
            self.pool = None
            if shortcut:
                self.downsample = Conv1D_BN(filters, 3, strides=2, padding='same', use_bias=use_bias)        
        else:
            strides = 1
            self.pool = Pooling(pool_type)
            if shortcut:
                self.downsample = Conv1D_BN(filters, 3, strides=2, padding='same', use_bias=use_bias)

        self.conv1 = layers.Conv1D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias, 
                                   kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=use_bias,
                                   kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()

        assert proj_type in ['identity', 'conv', None]
        if shortcut:
            if proj_type == 'conv':
                self.proj = Conv1D_BN(filters*2, 1, strides=1, padding='same', use_bias=use_bias)

            elif proj_type == 'identity':
                self.proj = ZeroPadding([int(filters // 2), filters - int(filters // 2)])

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.pool is not None:
            out = self.pool(out)

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(residual)
            out += residual

        out = tf.nn.relu(out)

        if self.proj_type is not None and self.shortcut:
            out = self.proj(out)

        return out

class VDCNN:
    def __init__(self, output_size=1, depth=9, vocab_size=None, input_length=None, 
                 embedding_size=16, shortcut=True, pool_type='max', proj_type='conv', use_bias=True, 
                 logits=True, patience=2, batch_size=50, repeate=10, epochs=1000,):
        super().__init__()

        self.name = "VDCNN"
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.embedding_size = embedding_size

        self.depth=depth
        self.shortcut = shortcut
        self.pool_type = pool_type
        self.proj_type = proj_type
        self.use_bias = use_bias
        self.logits = True

        self.n_blocks = {
            9: (1, 1, 1, 1), 
            17: (2, 2, 2, 2),
            29: (5, 5, 2, 2),
            49:(8, 8, 5, 3)
            }

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
    
        assert self.pool_type in ['max', 'k_max', 'conv']
        assert self.proj_type in ['conv', 'identity']

        self.model = tf.keras.Sequential(name=self.name)

        self.model.add(layers.Embedding(self.vocab_size, 
                                        self.embedding_size, 
                                        input_length=self.input_length))
        self.model.add(layers.Conv1D(filters=64, 
                                     kernel_size=3, 
                                     strides=1, 
                                     padding='same', 
                                     use_bias=self.use_bias, 
                                     kernel_initializer='he_normal'))

        # Convolutional Block 64
        for _ in range(self.n_blocks[self.depth][0] - 1):
            self.model.add(ConvBlock(64, 3, self.use_bias, self.shortcut))
        self.model.add(ConvBlock(64, 3, self.use_bias, self.shortcut, pool_type=self.pool_type, proj_type=self.proj_type))

        # Convolutional Block 128
        for _ in range(self.n_blocks[self.depth][1] - 1):
            self.model.add(ConvBlock(128, 3, self.use_bias, self.shortcut))
        self.model.add(ConvBlock(128, 3, self.use_bias, self.shortcut, pool_type=self.pool_type, proj_type=self.proj_type))

        # Convolutional Block 256
        for _ in range(self.n_blocks[self.depth][2] - 1):
            self.model.add(ConvBlock(256, 3, self.use_bias, self.shortcut))
        self.model.add(ConvBlock(256, 3, self.use_bias, self.shortcut, pool_type=self.pool_type, proj_type=self.proj_type))

        # Convolutional Block 512
        for _ in range(self.n_blocks[self.depth][3] - 1):
            self.model.add(ConvBlock(512, 3, self.use_bias, self.shortcut))
        self.model.add(ConvBlock(512, 3, self.use_bias, self.shortcut, pool_type=None, proj_type=None))

        self.model.add(KMaxPooling(k=8))
        self.model.add(layers.Flatten())

        # Dense layers
        self.model.add(layers.Dense(units=2048, activation='relu'))
        self.model.add(layers.Dense(units=2048, activation='relu'))
        self.model.add(layers.Dense(units=self.output_size, activation=self.activation))

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