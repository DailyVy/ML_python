import os.path

import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.models import load_model


def make_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=784, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# # 컨볼루션 신경망 설정
# def make_cnn():
#     model = Sequential()
#     model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu"))
#     model.add(MaxPooling2D(pool_size=2))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(128, activation="relu"))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation="softmax"))
#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#     return model


MY_EPOCH = 30
MY_BATCHSIZE = 200

def train(model, x, y):
    history = model.fit(x, y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE)
    model.save("./model/mlp.hd512.try11.h5")
    return history


if __name__ == "__main__":
    # MLP로 학습해보기
    (train_set, train_label), (test_set, test_label) = tf.keras.datasets.mnist.load_data("mnist.npz")

    train_set = train_set.reshape(60000, 784)
    test_set = test_set.reshape(10000, 784)

    print(train_label[0])  # 5

    # One-Hot-Encoding으로 바꿔주기
    train_label = tf.keras.utils.to_categorical(train_label, 10)
    print(train_label[0])  # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    test_label = tf.keras.utils.to_categorical(test_label, 10)

    # 데이터를 0에서 255사이의 값으로 변환하자
    # train_set 데이터를 바꿔주자
    train_set = train_set.astype('float64') / 255
    # test_set 데이터도 바꿔주자
    test_set = test_set.astype('float64') / 255

    mlp = make_model()
    train(mlp, train_set, train_label)
    acc = mlp.evaluate(test_set, test_label, batch_size=MY_BATCHSIZE)
    print(acc)

    # # 저장된 모델로 테스트하기
    # mlp = load_model("./model/mlp.hd512.h5")
    # acc = mlp.evaluate(test_set, test_label, batch_size=MY_BATCHSIZE)
    # print(acc)

    # # CNN 해보자 ==> 책 보고 따라했는데 안됨
    # (X_train, Y_train), (X_test,Y_test) = tf.keras.datasets.mnist.load_data("mnist.npz")
    # X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/255
    # X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/255
    # Y_train = np_utils.to_categorical(Y_train)
    # Y_test = np_utils.to_categorical(Y_test)
    # #
    # # # 컨볼루션 신경망 설정
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu"))
    # model.add(Conv2D(64, (3, 3), activation="relu"))
    # model.add(MaxPooling2D(pool_size=2))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation="relu"))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation="softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # # 모델 최적화 설정
    # MODEL_DIR = "./model/"
    # if not os.path.exists(MODEL_DIR):
    #     os.mkdir(MODEL_DIR)
    # modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
    # checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1, save_best_only=True)
    # early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)
    #
    # history = model.fit(train_set, train_label, validation_data=(test_set, test_label),
    #                   epochs=MY_EPOCH, batch_size=MY_BATCHSIZE, verbose=0,
    #                   callbacks=[early_stopping_callback, checkpointer])
    # acc = model.evaluate(test_set, test_label, batch_size=MY_BATCHSIZE)
    # print(acc)
