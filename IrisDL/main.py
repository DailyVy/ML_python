from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("Iris.csv")
    print(df.info()) # null값 없음
    print(df.head()) # Species 클래스, 즉 Y가 값이 아니고 문자열이네 ==> sklearn의 LabelEncoder
    # id 컬럼은 날려도 되잖아
    df = df.drop("Id", axis=1)
    print(df)
    print(df.corr())
    dataset = df.values
    print(dataset)

    # Train Set / Test Set 나누자 8:2로 나눠야징 150개니까 120/30
    # trainX = dataset[:120, :4]
    # testX = dataset[120:, :4]
    # 테스트셋이 죄다 Iris-Virginica 더라고
    X = dataset[:, :4]

    # Y는 일단 LabelEncoder를 하자 그 후 Train/Test나누자
    Y_obj = dataset[:, 4]

    # train set/ test set 을 split 해보자
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y_obj, test_size=0.2, shuffle=True, random_state=32)

    # Species를 숫자로 변환
    e = LabelEncoder()
    e.fit(train_Y)
    Y1 = e.transform(train_Y)
    Y_encoded = tf.keras.utils.to_categorical(Y1) # 원핫인코딩

    e.fit(test_Y)
    Y2 = e.transform(test_Y)
    test_Y = tf.keras.utils.to_categorical(Y2)


    # 모델을 min max 정규화 해보자
    scaler = MinMaxScaler()
    # trainX = scaler.fit_transform(trainX)
    # testX = scaler.fit_transform(testX)
    # print(trainX)

    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)

    print(train_X)
    print(test_X)

    # 모델 설정
    model = Sequential()
    model.add(Dense(16, input_dim=4, activation="relu")) # input_dim=4이다. feature가 4개니까~~~ SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm
    model.add(Dense(3, activation="softmax"))
    model.summary()

    # 모델 컴파일
    model.compile(loss="categorical_crossentropy", # 분류할거니까~
                  optimizer="adam",
                  metrics=["accuracy"])
    # fit합시당
    # model.fit(trainX, trainY, epochs=200, batch_size=20)
    # model.fit(train_X, Y_encoded, epochs=100, batch_size=10)
    # # evaluate합시당
    # print(model.evaluate(test_X, test_Y))
    #
    # # predict
    # print(model.predict(test_X))
    # print(test_Y)


    # 지금 데이터셋을 살펴보니 Iris-setosa, Iris-versicolor, Iris-virginica 로 정렬되어 있다.
    # 그러니까 데이터셋을 앞에서부터 120개로 자르면 Iris-virginica는 많이 안들어감... ㅜㅠ
    # sklearn.model_selection의 train_test_split 함수 사용함