from tensorflow.keras.layers import LSTM, Dense, GRU, Conv1D, Bidirectional
from tensorflow.keras import Sequential
from tensorflow import keras
import tensorflow as tf


class Model(keras.models.Model):
    def __init__(self, output_size=128):
        super(Model, self).__init__()
        self.output_size = output_size
    def build(self, input_shape=(500,3)):
        self.conv_layer = Conv1D(filters=1,kernel_size=3,strides=2)

        self.rnn_1f = LSTM(self.output_size, return_sequences=True,input_shape = input_shape)
        # self.rnn_1b = LSTM(self.output_size, return_sequences=True,input_shape = input_shape,activation='relu',go_backwards=True)
        # self.bidir_1 = Bidirectional(self.rnn_1f, backward_layer=self.rnn_1b, input_shape = input_shape)

        # self.rnn_2f = LSTM(int(self.output_size/2), return_sequences=True)
        # self.rnn_2b = LSTM(int(self.output_size/2), return_sequences=True,activation='relu',go_backwards=True)
        # self.bidir_2 = Bidirectional(self.rnn_2f, backward_layer=self.rnn_2b)     

        # self.rnn_3f = LSTM(int(self.output_size/4))
        # self.rnn_3b = LSTM(int(self.output_size/4),activation='relu',go_backwards=True)
        # self.bidir_3 = Bidirectional(self.rnn_3f, backward_layer=self.rnn_3b)

        # self.rnn_2 = LSTM(int(self.output_size/2), return_sequences=True)
        # self.rnn_3 = LSTM(int(self.output_size/4), return_sequences=True)
        self.output_layer = Dense(1, activation ='sigmoid')
        self.model = Sequential(
            [
                self.conv_layer,
                self.rnn_1f,
                #self.bidir_1,
                # self.bidir_2,
                #self.bidir_3,
                self.output_layer,
            ]
        )
        super(Model, self).build(input_shape)
        self.model.summary()
    
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