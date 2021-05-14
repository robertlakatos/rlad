import tensorflow as tf

class Simple:
    def __init__(self, vocab_size, input_lenght, embedding_size=300, hidden_layer_size=64, 
                 output_size=1, patience=2, batch_size=50, repeate=10, name="Simple"):
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.input_lenght = input_lenght
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.epochs = 1000
        self.patience = patience
        self.batch_size = batch_size
        self.repeate = repeate        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_train = None
        self.model = None
        self.history = []
        
        self.metrics = [tf.keras.metrics.BinaryAccuracy()]
        self.metrics_multi_label = False
        self.metrics_specificity = 0.5
        self.metrics_sensitivity = 0.5

        self.activation = "sigmoid"
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.monitor = "val_binary_accuracy"
        if self.output_size > 1:
            self.output_size = self.output_size + 1
            self.activation = "softmax"
            self.loss = tf.keras.losses.CategoricalCrossentropy()
            self.metrics[0] = tf.keras.metrics.CategoricalAccuracy()
            self.metrics_multi_label = True
            self.monitor="val_categorical_accuracy"

        # print(self.metrics_multi_label)
        self.metrics.append(tf.keras.metrics.AUC(multi_label=self.metrics_multi_label))
        self.metrics.append(tf.keras.metrics.Precision())
        self.metrics.append(tf.keras.metrics.Recall())
        self.metrics.append(tf.keras.metrics.TruePositives())
        self.metrics.append(tf.keras.metrics.TrueNegatives())
        self.metrics.append(tf.keras.metrics.FalsePositives())
        self.metrics.append(tf.keras.metrics.FalseNegatives())
        self.metrics.append(tf.keras.metrics.SensitivityAtSpecificity(self.metrics_specificity))
        self.metrics.append(tf.keras.metrics.SpecificityAtSensitivity(self.metrics_sensitivity))

        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor=self.monitor, 
                                                               patience=self.patience)
            
    def _create_model(self):
        self.model = tf.keras.Sequential(name=self.name)
        
        self.model.add(tf.keras.layers.Embedding(input_dim=self.vocab_size+1, 
                                                 output_dim=self.embedding_size, 
                                                 input_length=self.input_lenght))
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

    def set_data(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def get_history(self):
        return self.history

    def fit(self):
        self.history = []

        for i in range(self.repeate):            
            self._create_model()
            self.summary()
            history = self.model.fit(self.train_X,
                                     self.train_y,
                                     epochs=self.epochs,
                                     callbacks=[self.early_stopping],
                                     batch_size=self.batch_size,
                                     validation_data=(self.test_X, self.test_y))

            self.history.append(history)

        return self.get_history()