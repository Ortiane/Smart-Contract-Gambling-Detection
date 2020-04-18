from model import *
import argparse
import numpy as np
import tensorflow as tf
from preprocess import *
#import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data.csv", type=str)
    parser.add_argument("--output_size", default=1024, type=int)
    parser.add_argument("--max_seq_len", default=500, type=int)
    args = parser.parse_args()
    return args

def main():
    # Get arguments
    args = parse_args()
    DATA_FILE = args.data_file
    OUTPUT_SIZE = args.output_size
    MAX_SEQ_LEN = args.max_seq_len
    # Read in data
    x_train, y_train = process_data(DATA_FILE,MAX_SEQ_LEN)
    # Build and run model
    model = Model(
        output_size=OUTPUT_SIZE, 
    )
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    model.build(input_shape=(MAX_SEQ_LEN,3))
    #model.summary()
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=8,
        epochs=20, 
        verbose=1, 
        validation_split=0.1, 
        shuffle=True, 
        validation_freq=1
    )


if __name__ == '__main__':
    main()



