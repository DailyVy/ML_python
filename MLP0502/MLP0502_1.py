from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# XOR 해봅시다. ==> 퍼셉트론으로는 할 수 없음
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
# y = [0, 1, 1, 0] ==> scalar로 보는게 아니라 vector로 보는 거(각각 다른 클래스로 취급하고 싶음)
y = [[1, 0], [0, 1], [0, 1], [1, 0]]

if __name__ == "__main__":
    # clf 모델을 만들어보자
    clf = Sequential() # 입력부터 출력까지 Sequential 하니까
    # input을 2개받아 5개인 node를 만들 것이다.
    clf.add(Dense(500, input_dim=2, activation="relu"))
    clf.add(Dense(2, activation="softmax")) # 앞쪽 출력이 이번 입력이니까 input_dim=5인데 안적어줘도 된다!!
    clf.summary()
    # 모델 컴파일은 무조건 해줘야한다.
    # loss와 accuracy의 차이는?
    # 회귀문제는 binary_crossentropy 또는 MSE
    clf.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # 이제 학습합시다
    clf.fit(X, y, epochs=1000)
    # X 예측값을 봅시다.
    print(clf.predict(X))




# # 모델이 순차적으로 갈거니까 sequential입니다.
# model = Sequential()
# # layer를 추가해줄겁니다. convolution, maxpooling? 우리는 dense!
# # Dense layer를 사용할 것이다!
# #  맨 처음 layer에는 입력 개수를 명시해줘야 한다!
# #  Dense(2, input_dim=2) 출력이 두 개 니까(0 or 1) 2, 그 다음엔 입력은 2개 즉, input dimension은 2
# # kernel_initializer : weight값들을 뭘로 초기화 해줄건지 결정하는 옵션
# #  normal(정규화)도 있고 uniform, 포아송분포도 있고..
# # activation 함수도 넣어줘야해, softmax로 정규화
# #  softmax : 출력의 합이 1이되도록 만들려고
# model.add(Dense(2, input_dim=2, kernel_initializer="uniform", activation="softmax"))
# model.summary()
# # 모델 컴파일! 옵션을 지정해줘야 한다.
# # optimizer(loss optimizer) 잘 모르면 adam
# # loss함수는 분류문제라서 categorical_crossentropy, 만약 회귀라면? binary_crossentropy
# # metrics = ["accuracy"], 학습이 잘 됐는지 판단을 accuracy로 한다는 것!
# #  loss와 다르다! loss는 있을 수도 있다!!!!
# #  accuracy는 정답 맞으면 맞는 거! [0.9 0.1]이든 [0.54 0.46]이든 둘다 0인건 맞췄지
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# # 학습 데이터를 몇번 돌릴 건지 ==> epoch
# model.fit(X, y, epochs=200)
# # 모델 저장하는 건 다음 시간에 합시다.
# print(model.predict([[0, 0]])) # 2차원으로 학습시켰으니, test도 2차원으로 넣어줘!
# print(model.predict(X))

