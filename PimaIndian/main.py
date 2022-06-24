# 2교시
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  # 파일을 읽어들이기 위해
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import MinMaxScaler

# 1교시
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # 1교시
    # df = pd.read_csv("diabetes.csv")
    # df.columns = ["Pregnancies", "Glucose", "BP", "Skin", "Insulin", "BMI", "Pedigree", "Age", "Class"]
    # print(df)
    # print(df.shape) # 768, 9
    # print(df.head(5))
    # print()
    # print(df.info()) # null인게 없다!!! 넘 조아..
    # print()
    # print(df.describe())
    # print(df.corr())
    # print(df[["Pregnancies", "Outcome"]].corr()) # 두 컬럼의 상관도

    # print(df.value_counts("Pregnancies").sort_values()) # 클래스로 오름차순
    # print(df.value_counts("Pregnancies").sort_index()) # index로 정렬해서 value_counts()

    # 데이터프레임을 두 개로 쨌습니당. 클래스가 1인거랑 0인거랑(diabetes)
    # df_diabetes = df.loc[df["Outcome"] == 1]
    # df_normal = df.loc[df["Outcome"] == 0]
    #
    # print(df_diabetes.value_counts("Pregnancies").sort_index())
    # print(df_normal.value_counts("Pregnancies").sort_index())
    #
    # hist1 = df_diabetes.value_counts("Pregnancies").sort_index()
    # hist2 = df_normal.value_counts("Pregnancies").sort_index()
    # hist3 = hist1 / (hist1 + hist2)
    #
    # print(hist3)  # 14,15,17회는 모두 당뇨여서 df_normal에 존재X ==> NaN값이 생긴다.
    # print(hist3.fillna(1))

    # 이걸 Correlation으로 확인해볼래
    # DiabetesPedigreeFunction 컬럼 이름이 너무 길다. 컬럼 이름 변경하고 오세요. df.columns
    # print(df.corr())

    # heatmap 그래봅시다 - matplotlib, seaborn
    # plt.figure(figsize=(12, 12))
    # sns.heatmap(df.corr(), annot=True, cmap=plt.cm.PuRd, linewidths=0.1, linecolor="white")
    # plt.show()

    # FacetGrid (Seaborn)
    # grid = sns.FacetGrid(df, col="Class") # col에 Class를 주면 클래스별로 그래프가 나옴
    # grid.map(plt.hist, "Glucose", bins=10) # Glucose에 대한 histogram, 10개 구간으로(bins)
    # plt.show()

    # 2교시
    # numpy에서 txt파일을 읽어온다.

    # 2교시
    # 첫째줄에 String이 있어서 Error나므로 컬럼명을 지워준다.
    dataset = np.loadtxt("diabetes.csv", delimiter=",")
    print(dataset)

    # X = dataset[:, :8]
    # Y = dataset[:, 8]

    # 데이터를 쪼갭시다
    # Train set
    X = dataset[:700, :8]
    Y = dataset[:700, 8]
    # Test set
    testX = dataset[700:, :8]
    testY = dataset[700:, 8]

    # 정규화 하자 => sklearn 의 preprocessing
    # StandardScaler은 표준화를 쉽게 지원해 주는 함수입니다.
    # 다시 설명하자면 피처들을 평균이 0이고 분산이 1인 값으로 변환을 시켜줍니다
    
    # scaler = StandardScaler() # 이건 표준화!
    # scaler = Normalizer() # 이건 정규화

    # MinMaxScaler은 데이터의 값들을 0과 1사이의 범위 값으로 변환하여 줍니다.
    # 데이터들의 분포가 가우시안 분포가 아닐 경우에 적용을 해 볼 수 있습니다.
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X) # 정규화~
    testX = scaler.fit_transform(testX) # 테스트셋도 정규화 해줘야지

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    # model.summary()
    model.compile(loss="binary_crossentropy",  # 0~1 확률나오면 binary_crossentropy, 잘모르겠다 싶으면 mse
                  optimizer="adam",
                  metrics=["accuracy"])
    model.fit(X, Y, epochs=200, batch_size=50)
    # print(model.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])) # 맨 처음 사람꺼 예측해보자
    # 데이터셋을 Train/Test로 쪼개고 오시오
    # evaluation을 Test로 하자.
    print(model.evaluate(testX, testY))
