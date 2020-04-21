import csv
import numpy as np
import tensorflow as tf
from web3.auto import w3


def process_data(file_path, max_seq_len):
    x_train = []
    y_train = []
    with open(file_path) as f:
        csv_reader = csv.reader(f, delimiter=",")
        gambling = 0
        non_gambling = 0
        for i, row in enumerate(csv_reader):
            x = row[2:]
            y = int(row[0])
            # print(len(x))
            for ele in range(len(x)):
                x[ele] = float(w3.fromWei(int(x[ele]), "gwei"))
            x = np.array(x).reshape(-1, 3)
            # remove gambling contracts that are all zeros (you can't be gambling with no money)
            if not np.any(x) and y == 0:
                continue
            x_train.append(x)
            y_train.append(y)
            if y == 1:
                gambling += 1
            else:
                non_gambling += 1
        print(gambling, non_gambling)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        x_train, dtype="float64", maxlen=max_seq_len, padding="post", value=-1
    )

    y_train = np.array(y_train)
    return x_train, y_train


if __name__ == "__main__":
    process_data("data.csv")

