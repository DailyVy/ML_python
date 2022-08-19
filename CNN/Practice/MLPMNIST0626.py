from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf

"""Hyper Parameter"""
MY_EPOCH = 30
MY_BATCHSIZE = 200


def make_model():
    """
    model을 만듦 ==> Layer를 쌓음
    :return: model을 리턴
    """
    model = Sequential()
    model.add(Dense(512, input_dim=784, activation="relu")) # 784 = 28*28
    model.add(Dense(10, activation="softmax")) # softmax : 10개의 출력을 정규화(합이 1이 되도록) ==> 마치 확률처럼
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accruacy"])

    return model

def train(model, X, y):
    history = model.fit(X, y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE)
    model.save("./model/mlp_hd512_2.h5")

    return history

if __name__ == "__main__":

    # 데이터를 가지고 오자. 오프라인에서
    path = "../mnist.npz"
    (train_data, train_label), (test_data, test_label) = load_data(path)
    print(train_data.shape) # (60000, 28, 28)

    # data의 shape을 바꿔주자
    train_data = train_data.reshape(60000, 784)
    test_data = test_data.reshape(10000, 784)


    # label 을 one-hot-encoding
    train_label = tf.keras.utils.to_categorical(train_label, 10)
    test_label = tf.keras.utils.to_categorical(test_label, 10)

    # 모델 생성
    mlp = make_model()

    # 학습
    train(mlp, train_data, train_label)