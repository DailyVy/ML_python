from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# XOR 해봅시다
# Perceptron 으로는 불가하다.

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[1, 0], [0, 1], [0, 1], [1, 0]]

if __name__ == '__main__':
    # clf 모델을 만들어보자
    clf = Sequential() # 입력부터 출력까지 Sequential 하다
    # input을 두 개 받아 5개인 node를 만들 것이다.
    # clf.add(Dense(5, input_dim=2, activation="sigmoid")) # 0과 1사이의 값이 나온다.
    # loss가 발산해서 히든레이어의 노드를 늘려주었다. 그리고 activation 함수도 바꾸고
    clf.add(Dense(500, input_dim=2, activation="relu"))
    # 교수님 사부님 said:
    # 분류가 잘 안될때는 네가 설계한 네트워크가 분류할 만큼 충분한 파라미터를 가지고 있지 않아 그렇다.
    # 히든레이어의 노드를 늘려보면 학습이 될거다.
    # 앞 레이어의 출력이 뒷 레이어의 입력이 된다.
    clf.add(Dense(2, activation="softmax")) # 즉 input_dim=5이다. 생략가능, softmax는 확률값으로 나오게 하기 위해!
    # clf.summary()
    # 모델 컴파일은 무조건 해줘야 한다.
    clf.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # 이제 학습합시다.
    clf.fit(X, y, epochs=1000)
    # X 예측값을 봅시다
    print(clf.predict(X))

