from model import *
import argparse
import numpy as np
import tensorflow as tf
from preprocess import *
#import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data.csv", type=str)
    parser.add_argument("--output_size", default=512, type=int)
    parser.add_argument("--max_seq_len", default=100, type=int)
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
    print(f"Number of examples is {len(x_train)} ")
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
    model.summary()
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=20, 
        verbose=1, 
        validation_split=0.1, 
        shuffle=True, 
        validation_freq=1
    )
    # predict on the training set itself
    result = model.predict(
        x=x_train,
        verbose=0
    )
    size = len(y_train)
    accu_count = 0
    for (predict, true) in zip(result, y_train):
        final = 0
        if predict >= 0.5:
            final = 1
        if final == true:
            accu_count += 1
        print(f"Predict label {predict}, true label {true}")
    print(f"Accuracy is {accu_count/size}, benchmark is {1 - np.sum(y_train)/size}")


if __name__ == '__main__':
    main()



