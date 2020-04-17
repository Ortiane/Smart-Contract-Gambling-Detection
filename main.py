from model import Model
import argparse
import numpy as np
import tensorflow as tf
from preprocess import *
#import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data.csv", type=str)
    parser.add_argument("--output_size", default=128, type=int)
    parser.add_argument("--input_embedding", default=100000, type=int)
    parser.add_argument("--output_embedding", default=256, type=int)
    parser.add_argument("--max_seq_len", default=100, type=int)
    args = parser.parse_args()
    return args

def main():
    # Get arguments
    args = parse_args()
    DATA_FILE = args.data_file
    OUTPUT_SIZE = args.output_size
    INPUT_EMBEDDING = args.input_embedding
    OUTPUT_EMBEDDING = args.output_embedding
    MAX_SEQ_LEN = args.max_seq_len
    # DATA_FILE = "data.csv"
    # OUTPUT_SIZE = 200
    # INPUT_EMBEDDING = 1000
    # OUTPUT_EMBEDDING = 64
    # MAX_SEQ_LEN = 100
    # Read in data
    x_train, y_train = process_data(DATA_FILE, MAX_SEQ_LEN)

    # Build and run model
    model = Model(
        output_size=OUTPUT_SIZE, 
        input_embedding=INPUT_EMBEDDING, 
        output_embedding=OUTPUT_EMBEDDING,
    )
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    model.build(input_shape=(1,MAX_SEQ_LEN))
    model.summary()
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=4,
        epochs=20, 
        verbose=1, 
        validation_split=0.1, 
        shuffle=True, 
        validation_freq=1
    )


if __name__ == '__main__':
    main()