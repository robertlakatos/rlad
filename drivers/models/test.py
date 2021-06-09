from tensorflow.keras.layers import Input, Dense, LSTM, MaxPooling1D, Conv1D, Embedding
from tensorflow.keras.models import Model

input_layer = Input(shape=(400,))
emb = Embedding(input_dim=1000,
                output_dim=8,
                input_length=400)(input_layer)

conv1 = Conv1D(filters=32,
               kernel_size=8,
               strides=1,
               activation='relu',
               padding='same')(emb)
lstm1 = LSTM(32, return_sequences=True)(conv1)
output_layer = Dense(1, activation='sigmoid')(lstm1)
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()