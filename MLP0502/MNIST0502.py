import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# visualize해봅시다. matplotlib이용해서
# 이미지를 보여주는 함수 show_image(img)를 만들어보자
def show_image(img, label): # 이미지를 넘겨줄 것임 train_set[1] 뭐 이런식으로
    plt.imshow(255-img, cmap="gray")
    print(label)
    plt.show()

def show_data_value(labels):
    count_value = np.bincount(labels)  # pandas의 value_counts와 비슷
    print(count_value)
    # plt.hist(count_value)
    # plt.show()
    # 이상하다. 다시그려보자
    plt.bar(np.arange(0, 10), count_value)  # x축은 0~9, y축은 count_value
    plt.xticks(np.arange(0, 10))
    plt.grid()  # grid그려주자
    plt.show()

if __name__ == "__main__":
    path = "D:/ML_python/MLP0502/mnist.npz" # 파이썬에서 파일경로를 표현할 때 역슬래시는 X
    # 이렇게 하면 data set을 불러올 수 있음!
    (train_set, train_label), (test_set, test_label) =\
        tf.keras.datasets.mnist.load_data(path)

    # print(train_set.shape) # (60000, 28, 28) 3차원 배열 28x28이 60000개
    # print(train_set[0].shape) # (28, 28) [0~ 59999]
    # print(train_label.shape) # (60000,) 값은 60000개인데 답만 있기 때문에
    # print(train_label[:10]) # [5 0 4 1 9 2 1 3 1 4]
    # print(train_set[0]) # 첫번째 이미지
    # print(train_set[0][10]) # 첫번째 행
    # print(train_set[0][0][0]) # 첫번째 행의 첫번째 값

    # 이미지를 봅시다 show_image
    # show_image(train_set[0])
    # show_image(train_set[1])
    # => 계속해서 볼 수 있는 코드를 만든다.
    # user로 부터 입력을 받아서 이미지 띄우기 -> 아무 입력이나 주면 다음 이미지로 넘어가고 q 넣어주면 끝나는 걸로!
    userInput = "q" # ""였는데 그림 안보려고 "q"로 적어놓음
    index = 0

    while userInput != "q": # input이 q가 들어오면 끝나는 거야
        show_image(train_set[index], train_label[index])
        index += 1 # 이제 그다음 거 봐야지
        userInput = input("next? : ") # 이제 input을 받자

    # 데이터의 분포를 확인해봅시다.
    # 정답을 보면 이미지가 몇 개 있는지 알 수 있다.
    # label의 타입을 보자
    # print(type(train_label)) # <class 'numpy.ndarray'>

    # count_value = np.bincount(train_label) # pandas의 value_counts와 비슷
    # print(count_value)
    # # plt.hist(count_value)
    # # plt.show()
    # # 이상하다. 다시그려보자
    # plt.bar(np.arange(0, 10), count_value) # x축은 0~9, y축은 count_value
    # plt.xticks(np.arange(0, 10))
    # plt.grid() # grid그려주자
    # plt.show()
    # 위 코드를 함수로 구현해보자 show_data_values
    # show_data_value(train_label)
    # show_data_value(test_label)

    train_set = train_set.reshape(len(train_set), 784) # 28x28 = 784 (60000, 784)
    test_set = test_set.reshape(len(test_set), 784)
    # 학습할 때만 쓴당
    # clf = RandomForestClassifier()
    # clf.fit(train_set, train_label)
    # joblib.dump(clf, "rf_mnist.pkl") # 모델과 모델이름 저장하기

    # 이제 학습했으니까 fit안하고 load하기
    clf = joblib.load("rf_mnist.pkl") # 파일명을 인자로 준다.
    # 가져왔으니 테스트하자 => 가져온거라서 얘가 RandomForest인지 몰라.. predict그냥 직접 쳐줘
    # print(clf.predict(train_set[0:1]))

    print(clf.score(test_set, test_label)) # 모델의 출력이랑 정답을 비교해서 score를 계산해준다.
