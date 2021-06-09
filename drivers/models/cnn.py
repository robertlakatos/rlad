import tensorflow as tf

class CNN:
    def __init__(self, vocab_size=None, input_length=None, embedding_size=300, hidden_layer_size=64, 
                 output_size=1, patience=2, batch_size=50, repeate=10, epochs=1000):
        super().__init__()

        self.name = "CNN"

        self.vocab_size = vocab_size
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.hidden_layer_size = hidden_layer_size
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
                                                 
        self.model.add(tf.keras.layers.Flatten())
        
        self.model.add(tf.keras.layers.Dense(units=self.hidden_layer_size))
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