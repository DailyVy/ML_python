from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.datasets.mnist import load_data
import tensorflow as tf

if __name__ == "__main__":
    # 모델을 만들자
    model = Sequential()
    # filter 한개만 쓰자(예제는 32개 였지만)
    # kernel_size는 5x5
    # strides는 filter sliding을 어떻게 할건가 1x1이라면 x축 방향으로 1칸, y축 방향으로 1칸
    # padding : zero padding
    model.add(Conv2D(filters=32,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     activation='relu',
                     input_shape=(28, 28, 1), # 28x28x1 : 흑백이라 채널은 1!
                     padding="same")) # padding 옵션 : filter 처리 후 원본 사이즈와 똑같도록!
    # 이상태에서 model summary 해보자
    # model.summary() # parameter가 26개 밖에 없음 5x5 + 1(bias) -> filter수 늘어날 수록 (25+1) * n 이넹
    # Maxpool
    # 영상을 반으로 줄이고 최대값만 남기겠다. ==> stride를 2로 준게 반으로 줄인 것~
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2))) # weight없음, 즉 weight가 늘어나지 않음
    # 이제 14x14가 됐는데 여기서 또 convolution
    #  filters=1 안하고 1만 해도 됨, 제일 앞에 있는게 filters라서 그래
    #  마치 Dense에서의 출력층 처럼
    #  strides = (1, 1) 생략가능. default가 이거라서
    #  MaxPool layer가 입력층이라 input_shape 생략해도 된다.
    model.add(Conv2D(64, kernel_size=(5, 5), activation="relu", padding="same")) # 여기도 weight는 26개
    # 또 MaxPool 합시다.
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2))) # 최종적으로 이미지는 7x7x1
    # 7 x 7 로 줄면서 중요한 특징만을 남겨둔 채로...

    # 이제 이걸 신경망에 넣을 때 flatten을 하는 것 ==> 입력층
    model.add(Flatten()) # (None, 49) 7x7을 펴니까 49개
    # 이제 신경망 모델에 이거 넣자
    model.add(Dense(1000, activation='relu')) # 100개의 출력...(49 + 1)*100 = 5000개의 weight
    model.add(Dense(10, activation='softmax')) # 이제 최종 출력 합시당. 총 6062개의 weight 이 중 62개는 conv층

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    # 데이터를 가지고 옵시다
    path = 'mnist.npz'
    (train_data, train_label), (test_data, test_label) = load_data(path)
    # print(train_data.shape) # 60000, 28, 28

    # 이거도 reshape 해줘야 한다. ==> 채널 수 때문에!!
    # convolution 할 때는 채널을 명확하게 써줘야 한다.
    train_data = train_data.reshape(60000, 28, 28, 1) # 채널수를 꼭 적어줘야해
    train_label = tf.keras.utils.to_categorical(train_label, 10) # one-hot-encoding

    model.fit(train_data, train_label, epochs=10, batch_size=200)
    # 40만개의 weight, 6000개의 weight인데 정확도가 엇비슷..

    model.save("cnn_mlp_practice.h5")