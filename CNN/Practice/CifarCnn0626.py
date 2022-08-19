from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
import tensorflow as tf



(train_data, train_label), (test_data, test_label) = load_model("cifar-10-batches-py/test_batch")

if __name__ == "__main__":
    print(train_data.shape)