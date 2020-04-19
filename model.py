from tensorflow.keras.layers import LSTM, Dense, GRU, Conv1D, Bidirectional, Embedding
from tensorflow.keras import Sequential
from tensorflow import keras
import tensorflow as tf


class Model(keras.models.Model):
    def __init__(self, output_size=128):
        super(Model, self).__init__()
        self.output_size = output_size
    def build(self, input_shape=(10000,3)):
        self.conv_layer = Conv1D(input_shape=input_shape, filters=256, kernel_size=3, strides=2, activation='relu')
        self.rnn_1 = GRU(self.output_size,return_sequences=True)
        self.rnn_2 = GRU(self.output_size,return_sequences=True)
        self.rnn_3 = GRU(self.output_size)
        #self.rnn_4 = LSTM(self.output_size)


        # self.rnn_1f = LSTM(self.output_size, return_sequences=True)
        # self.rnn_1b = LSTM(self.output_size, return_sequences=True,activation='relu',go_backwards=True)
        # self.bidir_1 = Bidirectional(self.rnn_1f, backward_layer=self.rnn_1b)

        # self.rnn_2f = LSTM(int(self.output_size/2), return_sequences=True)
        # self.rnn_2b = LSTM(int(self.output_size/2), return_sequences=True,activation='relu',go_backwards=True)
        # self.bidir_2 = Bidirectional(self.rnn_2f, backward_layer=self.rnn_2b)     

        # self.rnn_3f = LSTM(int(self.output_size/4))
        # self.rnn_3b = LSTM(int(self.output_size/4),activation='relu',go_backwards=True)
        # self.bidir_3 = Bidirectional(self.rnn_3f, backward_layer=self.rnn_3b)

        # self.rnn_2 = LSTM(int(self.output_size/2), return_sequences=True)
        # self.rnn_3 = LSTM(int(self.output_size/4), return_sequences=True)
        #self.dense_layer = Dense(128, activation = 'sigmoid')
        self.output_layer = Dense(1, activation ='sigmoid')
        self.model = Sequential(
            [
                #self.embedding,
                self.conv_layer,
                self.rnn_1,
                self.rnn_2,
                self.rnn_3,
                #self.rnn_4,
                # self.bidir_1,
                # self.bidir_2,
                # self.bidir_3,
                #self.dense_layer,
                self.output_layer,
            ]
        )
        super(Model, self).build(input_shape)
        #self.model.summary()
    
    def call(self, inputs):
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, 0)
        return self.model(inputs)


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