import tensorflow as tf

class RNN:
    def __init__(self, vocab_size=None, input_length=None, embedding_size=300, hidden_layer_size=64, output_size=1, 
                patience=2, batch_size=50, repeate=10, epochs = 1000, 
                name="LSTM", layers=[{"activation":"tanh", "dropout" : 0.0}]):
        super().__init__()

        self.name = name.upper()
        self.layers=layers

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
        self.X_test = None
        self.y_train = None
        self.model = None
        self.history = []
        
        self.metrics = ["accuracy"]
        self.metrics_multi_label = False
        self.metrics_specificity = 0.5
        self.metrics_sensitivity = 0.5
        
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.monitor = "val_loss" # "val_binary_accuracy"

        self.activation_out = "sigmoid"
        if self.output_size > 1:
            self.activation_out = "softmax"

        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor=self.monitor, 
                                                               patience=self.patience)

    def create_model(self):
        self.model = tf.keras.Sequential(name=self.name)

        self.model.add(tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size, input_length=self.input_length))
        
        if self.name == "LSTM":
            for layer in self.layers[:-1]:
                self.model.add(tf.keras.layers.LSTM(units=self.embedding_size, 
                                                    activation=layer["activation"],
                                                    recurrent_dropout=layer["dropout"],
                                                    return_sequences=True))

            self.model.add(tf.keras.layers.LSTM(units=self.embedding_size, 
                                                activation=self.layers[-1]["activation"],
                                                recurrent_dropout=self.layers[-1]["dropout"]))
        elif self.name == "GRU":
            for layer in self.layers[:-1]:
                self.model.add(tf.keras.layers.GRU(units=self.embedding_size, 
                                                    activation=layer["activation"],
                                                    recurrent_dropout=layer["dropout"],
                                                    return_sequences=True))

            self.model.add(tf.keras.layers.GRU(units=self.embedding_size, 
                                                activation=self.layers[-1]["activation"],
                                                recurrent_dropout=self.layers[-1]["dropout"]))
        else:
            for layer in self.layers[:-1]:
                self.model.add(tf.keras.layers.LSTM(units=self.embedding_size, 
                                                    activation=layer["activation"],
                                                    recurrent_dropout=layer["dropout"],
                                                    return_sequences=True))

            self.model.add(tf.keras.layers.LSTM(units=self.embedding_size, 
                                                activation=self.layers[-1]["activation"],
                                                recurrent_dropout=self.layers[-1]["dropout"]))
        
        self.model.add(tf.keras.layers.Dense(units=self.output_size, 
                                             activation=self.activation_out))

        self.model.compile(optimizer='adam',
                           loss=self.loss,
                           metrics=self.metrics)

    def summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            print("Model is not initialized")

    def set_data(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def get_history(self):
        return self.history

    def fit(self, train_X, train_y, val_X, val_y, summary=False):
        self.create_model()

        if summary == True:
            self.summary()

        self.history = self.model.fit(train_X, 
                                      train_y, 
                                      epochs=self.epochs, 
                                      callbacks=self.early_stopping, 
                                      batch_size=self.batch_size, 
                                      validation_data=(val_X, val_y))

        return self.history

    def __call__(self, inputs):
        return self.model(inputs)