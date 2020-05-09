from tensorflow.keras.layers import LSTM, Dense, GRU, Conv1D, Bidirectional, Embedding, Masking
from tensorflow.keras import Sequential
from tensorflow import keras
import tensorflow as tf


class Model(keras.models.Model):

    def __init__(self, output_size=128, num_filters=8, num_layers=2):
        super(Model, self).__init__()
        self.output_size = output_size
        self.num_filters = num_filters
        self.num_layers = num_layers

    def build(self, input_shape=(10000,3)):
        self.mask = Masking(-1)
        self.conv_layer = Conv1D(input_shape=input_shape, filters=self.num_filters, kernel_size=3, strides=3, activation='relu')

        lstm_list = []
        for n in range(self.num_layers - 1):
            lstm_list.append(Bidirectional(LSTM(self.output_size, return_sequences=True)))

        lstm_list.append(Bidirectional(LSTM(self.output_size)))

        self.lstm = Sequential(lstm_list)
        self.output_layer = Dense(1, activation ='sigmoid')
        super(Model, self).build(input_shape)
    
    def call(self, inputs):
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, 0)
        x = self.mask(inputs)
        x = self.conv_layer(x)
        x = self.lstm(x)
        x = self.output_layer(x)
        return x


if __name__ == '__main__':    
    # Build and run model
    MAX_SEQ_LEN = 100
    OUTPUT_SIZE = 256
    model = Model(
        output_size=OUTPUT_SIZE, 
    )
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    model.build()
    model.summary()