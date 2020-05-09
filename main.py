import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from model import *
from preprocess import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# tf.config.experimental.set_visible_devices([], 'GPU')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data.csv", type=str)
    parser.add_argument("--output_size", default=128, type=int)
    parser.add_argument("--num_filters", default=8, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--max_seq_len", default=100, type=int)
    args = parser.parse_args()
    return args


def main():
    # Get arguments
    args = parse_args()
    DATA_FILE = args.data_file
    OUTPUT_SIZE = args.output_size
    NUM_FILTERS = args.num_filters
    NUM_LAYERS = args.num_layers
    MAX_SEQ_LEN = args.max_seq_len
    # Read in data
    X, y = process_data(DATA_FILE, MAX_SEQ_LEN)
    print(f"Number of examples is {len(X)} ")
    fig = plt.figure(1, figsize=(8.78, 6.7))
    # Build and run model
    model = Model(
        output_size=OUTPUT_SIZE, num_filters=NUM_FILTERS, num_layers=NUM_LAYERS
    )
    model.compile(loss="binary_crossentropy", optimizer="adagrad", metrics=["accuracy"])
    model.build(input_shape=(MAX_SEQ_LEN, 3))
    model.summary()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=18
    )

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=32,
        epochs=25,
        verbose=1,
        validation_data=(X_test, y_test),
        shuffle=True,
        validation_freq=1,
    )

    # https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
    y_pred = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_value = auc(fpr, tpr)

    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label=f"LSTM (area = {auc_value:.3f})")

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("ROC_curve.png")


if __name__ == "__main__":
    main()
