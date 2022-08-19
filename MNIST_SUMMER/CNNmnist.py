# 1. MNIST CNN 모델 만들기
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def make_model():
    # Make model
    model = Sequential()
    # Layer 1
    model.add(Conv2D(filters=32,
                     kernel_size=(5, 5), # 5*5 =25 , (25+1(bias)) * 32(filter) = 832 => weight
                     strides=(1, 1), # filter sliding 을 x, y축 각 방향으로 한 칸씩
                     activation="relu",
                     input_shape=(28, 28, 1), # 28x28x1 : 흑백이라서 채널은 1
                     padding="same")) # filter 처리 후 원본 사이즈와 같도록
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2))) # (2, 2) maxpool은 2x2에서 maximum 값을 찾는다는 것
    # Layer 2
    model.add(Conv2D(filters=64, # filters= 생략 가능
                     kernel_size=(5, 5),
                     # strides=(1, 1), # default 생략 가능
                     activation="relu",
                     padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2))) # 7x7x64 = 3136
    model.add(Dropout(0.25)) # 과적합 방지를 위한 Dropout : Hidden layer의 node 증 일부를 임의로 꺼주는 것
    # Flatten
    model.add(Flatten()) # Fully Connected Layer 생성
    # input layer
    model.add(Dense(100, activation="relu")) # 313700 개의 weight
    model.add(Dropout(0.5)) # Dropout
    model.add(Dense(10, activation="softmax"))

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def make_model_everyDL(): # 모두의 딥러닝 버전
    # Make model
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=(1, 1),  # filter sliding 을 x, y축 각 방향으로 한 칸씩
                     activation="relu",
                     input_shape=(28, 28, 1)))  # 28x28x1 : 흑백이라서 채널은 1
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())  # Fully Connected Layer 생성
    # input layer
    model.add(Dense(128, activation="relu"))  # 313700 개의 weight
    model.add(Dropout(0.5))  # Dropout
    model.add(Dense(10, activation="softmax"))

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def data_preprocessing(data, label):
    print(data.shape[0], data.shape)
    return_data = data.reshape(data.shape[0], 28, 28, 1).astype("float32") / 255
    return_label = tf.keras.utils.to_categorical(label, 10)
    return return_data, return_label

if __name__ == "__main__":
    # 데이터를 가지고 오자
    (train_data, train_label), (test_data, test_label) = load_data()

    # print(train_data.shape) # (60000, 28, 28)

    ######################################################################## data reshape
    # convolution 시에는 채널을 명확하게 써줘야 한다.
    # 1. 수업시간
    # train_data = train_data.reshape(60000, 28, 28, 1)
    # train_label = tf.keras.utils.to_categorical(train_label, 10) # one-hot-encoding
    # test_data = test_data.reshape(10000, 28, 28, 1)
    # test_label = tf.keras.utils.to_categorical(test_label, 10)

    # 2. 모두의 딥러닝
    train_data, train_label = data_preprocessing(train_data, train_label)
    test_data, test_label = data_preprocessing(test_data, test_label)

    ######################################################################## end of data reshape

    ############################################################ Train and Evaluate : ML시간
    # model = make_model()
    # model.fit(train_data, train_label, epochs=30, batch_size=200)
    # model.save("./model/cnn_mlp_e30_b200_dropout.h5")

    # model1 = load_model("./model/cnn_mlp_e30_b200.h5")
    # model2 = load_model("./model/cnn_mlp_e30_b200_dropout.h5")

    # acc1 = model1.evaluate(test_data, test_label)
    # acc2 = model2.evaluate(test_data, test_label)

    # print(acc1) # [0.09579916298389435, 0.984499990940094]
    # print(acc2) # [0.03040432184934616, 0.9926000237464905]

    ############################################################ Train and Evaluate : 모두의 딥러닝
    # model = make_model_everyDL()  # 2. 모두의 딥러닝 ver

    # 모델 최적화 설정
    # modelpath = "./model/MNIST_CNN.hdf5"
    # checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", verbose=1, save_best_only=True)
    # early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)
    """
    ModelCheckpoint() : 학습 중인 모델을 저장하는 함수,
        verbose=1 : 모델이 저장될 곳을 정하고 진행되는 현황을 모니터할 수 있도록 verbose=1(True) 
        monitor="val_loss" : 학습된 모델을 검증셋에 적용해 얻은 오차 
                            (history.history에 있음 : loss, accuracy, val_loss, val_accruacy)
        save_best_only=True : 최고의 모델 하나만 저장하기
        
    EarlyStopping() : 학습이 진행되어도 테스트셋 오차가 줄어들지 않으면 학습을 자동으로 멈추게 하는 함수
        monitor : model.fit()의 실행 결과 중 어떤 것을 이용할 지 정하는 것
        patience : patience = 10 이면 검증셋의 오차가 10번 이상 낮아지지 않을 경우 학습 종료
    
    => 이 둘을 사용하여 최적의 모델을 저장
    """

    # 모델 실행
    # history = model.fit(train_data, train_label,
    #                     validation_split=0.25, epochs=30, batch_size=200, verbose=0,
    #                     callbacks=[early_stopping_callback, checkpointer])

    # 테스트 정확도 출력
    # print("\n Test Accuracy : %.4f" % (model.evaluate(test_data, test_label)[1]))

    # 검증셋과 학습셋의 오차 저장
    # y_vloss = history.history["val_loss"]
    # y_loss = history.history["loss"]

    # 그래프로 표현
    # x_len = np.arange(len(y_loss))
    # plt.plot(x_len, y_vloss, marker=".", c="red", label="Testset_loss")
    # plt.plot(x_len, y_loss, marker=".", c="blue", label="Trainset_loss")

    # 그래프에 그리드를 주고 레이블을 표시
    # plt.legend(loc="upper right")
    # plt.grid()
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.show()

    ################################################### 모델 테스트 해볼래
    # model = load_model("./model/MNIST_CNN.hdf5")
    # predict = model.predict(test_data)
    # print(predict[0].argmax(), test_label[0].argmax())

    # for i in range(10):
    #     print(predict[i].argmax(), test_label[i].argmax())