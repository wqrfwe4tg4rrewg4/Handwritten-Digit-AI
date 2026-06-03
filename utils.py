import numpy as np
from tensorflow import keras

def load_dataset():
    print("Загружаю MNIST...")

    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0

    # one-hot
    y_train_onehot = np.zeros((len(y_train), 10))
    y_train_onehot[np.arange(len(y_train)), y_train] = 1

    return x_train, y_train_onehot

CLASSES = 10