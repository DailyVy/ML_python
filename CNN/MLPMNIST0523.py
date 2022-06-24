from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf

if __name__ == "__main__":
    # 모델을 만들자
    model = Sequential()
    model.add(Dense(512, input_dim=784, activation='relu'))
    model.add(Dense(10, activation='softmax')) # softmax를 하는 이유? : 10개의 출력을 정규화(합이 1이 되도록) ==> 확률처럼
    model.summary()

    model.compile(loss="categorical_crossentropy",
                  optimizer='adam',
                  metrics=['accuracy']) # 성능평가는 accruracy

    # 데이터를 가지고 옵시다
    path = 'mnist.npz'
    (train_data, train_label), (test_data, test_label) = load_data(path)
    print(train_data.shape) # 60000, 28, 28

    # data의 shape를 바꿔주자
    train_data = train_data.reshape(60000, 784)
    test_data = test_data.reshape(10000, 784)

    # label을 one-hot-encoding! 카테고리를 나누는 형태로!! scalar로 보는게 아니라
    # tf.keras.utils.to_categorical
    train_label = tf.keras.utils.to_categorical(train_label, 10)
    test_label = tf.keras.utils.to_categorical(test_label, 10)

    # x, y 다 만들었으면 fit하면 된다.
    model.fit(train_data, train_label, epochs=10, batch_size=200)
    # 모델 저장하자
    # model.save("mlp_0523_1.h5")