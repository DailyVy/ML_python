import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

    mlp = Sequential()
    # layer를 쌓읍시다.
    mlp.add(Dense(512, input_dim=784, activation="relu"))
    mlp.add(Dense(10, activation="softmax"))
    # 어떻게 생겼는지 봅시다.
    mlp.summary()
    # 컴파일합시다.
    mlp.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # 학습합시다.
    mlp.fit(train_set, train_label, epochs=30, batch_size=200) # batch가 200개니까 300번 돌아가는거지 (데이터셋 총 60000개)
    # 결과값을 봅시다.(평가)
    print(mlp.evaluate(test_set, test_label, batch_size=200)) # batch_size 안 넣어도 되는데 넣는게 좀 더 빨라
    # [0.7706162333488464, 0.9757999777793884]
    # qna. 이 값이 무엇을 의미하는 거지? -> loss and accuracy

    # 학습 모델 저장해보자
    mlp.save("mlp_hd512_e30.h5") # hidden node가 512, epoch이 30개인 확장자가 h5 파일


