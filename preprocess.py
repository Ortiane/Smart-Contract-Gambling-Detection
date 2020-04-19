import csv
import numpy as np
import tensorflow as tf
from web3.auto import w3

def process_data(file_path, max_seq_len):
    x_train = []
    y_train = []
    with open(file_path) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(csv_reader):
            x = row[2:]
            #print(len(x))
            for ele in range(len(x)):
                x[ele] = float(w3.fromWei(int(x[ele]),'gwei'))
            x = np.array(x).reshape(-1,3)
            #print(x.shape)
            x_train.append(x)
            y_train.append(int(row[0]))
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
                    x_train,
                    dtype='float64',
                    maxlen=max_seq_len, 
                    padding='post', 
                    truncating='post',
                    value=-1.0
                )
    # x_train = tf.convert_to_tensor(x_train,dtype=tf.float64)
    # y_train = tf.convert_to_tensor(y_train,dtype=tf.float64)
    
    y_train = np.array(y_train)
    #print(x_train.shape)
    return x_train, y_train

if __name__ == '__main__':
    process_data('data.csv')