from tensorflow.keras.layers import LSTM, Dense, GRU, Embedding, Dropout
from tensorflow.keras import Sequential
from tensorflow import keras
import tensorflow as tf


class Model(keras.models.Model):
    def __init__(self, output_size=128, input_embedding=100000, output_embedding=64):
        super(Model, self).__init__()
        self.output_size = output_size
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding

    def build(self, input_shape=(1,100)):
        self.embedding = Embedding(self.input_embedding, self.output_embedding)
        self.recurr_unit = LSTM(self.output_size)
        self.dropout_layer_1 = Dropout(0.2)
        self.dropout_layer_2 = Dropout(0.2)
        self.output_layer = Dense(1, activation ='sigmoid')
        self.model = Sequential(
            [
                self.embedding,
                self.dropout_layer_1,
                self.recurr_unit,
                self.dropout_layer_2,
                self.output_layer,
            ]
        )
        super(Model, self).build(input_shape)
    
    def call(self, inputs):
        return self.model(inputs)
