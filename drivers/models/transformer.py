# https://arxiv.org/pdf/1706.03762.pdf
# Encoder part
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention
                                    )
        return output

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"),
             Dense(embed_dim), ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        
        self.token_emb = Embedding(input_dim=vocab_size, 
                                   output_dim=embed_dim)

        self.pos_emb = Embedding(input_dim=maxlen, 
                                 output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class Transformer:
    def __init__(self, vocab_size=None, input_length=None, embedding_size=512,
                 num_heads=8, ff_dim=2048, N=6, dropout=0.1, train_len=None,
                 output_size=1, patience=2, batch_size=50, repeate=10, epochs=1000):
        super().__init__()

        self.name = "Transformer"
        self.vocab_size = vocab_size
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.N = N
        self.train_len = train_len
        self.warmup_steps = 4000
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
        embedding_layer = TokenAndPositionEmbedding(self.input_length, 
                                                    self.vocab_size, 
                                                    self.embedding_size)
        x = embedding_layer(inputs)

        for i in range(0, self.N):
            transformer_block = TransformerBlock(self.embedding_size, 
                                                 self.num_heads, 
                                                 self.ff_dim)
            x = transformer_block(x)

        x = Flatten()(x)

        x = Dropout(self.dropout)(x)

        outputs = Dense(self.output_size, activation=self.activation)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        step_num = self.train_len // self.batch_size
    
        lrate = (self.embedding_size ** -0.5) * min([step_num ** -0.5, step_num * (self.warmup_steps ** -1.5)])
        
        optimizer = tf.keras.optimizers.Adam(beta_1=0.9,
                                        beta_2=0.98,
                                        learning_rate=lrate,
                                        epsilon=1e-09)

        self.model.compile(optimizer=optimizer,
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