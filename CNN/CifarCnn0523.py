# import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
import tensorflow as tf

def make_model():
    model = Sequential()
    model.add(Conv2D(filters=32, # 채널 3개에 bias하나씩 주는게 아니라 채널 3개 전체에 bias 하나씩 주더라
                     kernel_size=(5, 5), # 채널 3개에 filter하나라면 weight는 25 + 25 + 25 + 1 = 76 ==> 그래서 filter가 32개라면 76 * 32 라서 2,432의 weight가 나온다.
                     strides=(1, 1),
                     activation='relu',
                     input_shape=(32, 32, 3),  # 32x32x3 : 채널 3개
                     padding="same"))  # padding 옵션 : filter 처리 후 원본 사이즈와 똑같도록!
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation="relu", padding="same"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # 최종적으로 이미지는 7x7x1
    model.add(Flatten())  # (None, 49) 7x7을 펴니까 49개
    # 이제 신경망 모델에 이거 넣자
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model

# HyperParameter
MY_EPOCH = 10
MY_BATCHSIZE = 200
filename = f'./model/cnn_e({MY_EPOCH}).h5'

def train(model, x, y):
    x = x / 255.
    y = tf.keras.utils.to_categorical(y, 10)
    history = model.fit(x, y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE)
    model.save(filename)

    return history

def test_all(x, y):
    model = load_model(filename)
    x = x / 255.
    y = tf.keras.utils.to_categorical(y, 10)
    test_loss, test_acc = model.evaluate(x, y)

# 데이터를 가지고 옵시다
(train_data, train_label), (test_data, test_label) = cifar10.load_data()

if __name__ == "__main__":
    print(train_data.shape) # (50000, 32, 32, 3)
    print(test_data.shape) # (10000, 32, 32, 3)

    class_names = ['airplane', 'automobile', ' bird', 'car', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 그림 봅시당
    # for i in range(5):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(train_data[i])
    #     plt.xlabel(class_names[train_label[i][0]])
    # plt.show()

    # 모델을 만들자 ==> make_model() 함수로
    cnn = make_model()
    # 학습합시다 ==> train() 함수
    train(cnn, train_data, train_label)
    # 테스트합시다 ==> test_all() 함수
    test_all(test_data, test_label)

