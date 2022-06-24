import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

###### ageVar() : age 결측치 채우는 법을 바꿨다 ==> 62 %... 별루야
##### ageVar2() : 결측치 샘플 제우고, age가... float인게 있어 ==> outlier 제거, 54 %...
###### 가만보니 dropna() 했던건 대부분 정답율이 떨어진다.

# 일단 생존율을 한 번 확인해봅시다.
def show_group_rate(feature):
    df_survive = df_train.loc[df_train["Survived"] == 1]
    df_dead = df_train.loc[df_train["Survived"] == 0]

    sur_info = df_survive[feature].value_counts(sort=False)
    dead_info = df_dead[feature].value_counts(sort=False)

    fig = plt.figure()
    plt.title("Survival Rate of {0}".format(feature))

    for i, index in enumerate(sur_info.index):
        fig.add_subplot(1, len(sur_info), i+1)
        plt.pie([sur_info[index], dead_info[index]], labels=["Survived","Dead"], autopct="%0.1f %%")
        plt.title("Survival Rate of {0}".format(index))

    plt.show()

def ageVar():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    pId = df_test["PassengerId"]

    # show_group_rate("Pclass")
    # show_group_rate("Sex")
    # show_group_rate("SibSp")
    # show_group_rate("Parch")
    #
    # print(df_train.describe())
    # print(df_train.info())
    # print(df_train.corr())

    # Age - histogram
    # df_train["Age"].hist(bins=20, figsize=(18,8))
    # plt.show()

    # 상관도를 보자
    # fig = plt.figure(figsize=(10,10))
    # sns.heatmap(df_train.corr(), linewidths=0.01, square=True, annot=True, cmap=plt.cm.viridis, linecolor="white")
    # plt.title("Correlation between features")
    # plt.show()

    ############### 데이터 전처리

    # 필요없는 컬럼 지우기
    df_train.drop(["PassengerId", "Fare", "Cabin", "Ticket", "Name"],axis=1,inplace=True)

    # 난 일단 Age를 10단위로 묶어보고 싶어
    # df_train["Age"].fillna(df_train["Age"].mean(), inplace=True) # 결측치를 평균으로 채워줌
    #
    # for i in range(len(df_train)):
    #     age = int(df_train.loc[i, "Age"] / 10)
    #     df_train.loc[i,"Age"] = age
    #
    # df_train["Age"] = df_train["Age"].map({0:"Kids", 1:"Teen", 2:"20s", 3:"30s",
    #                                        4:"40s", 5:"50s", 6:"60s", 7:"Old", 8:"Old"})
    # print(df_train["Age"].head(10))
    # show_group_rate("Age")


    # Age를 평균치 말고 다른 값으로 넣어주면 어떨까? Age와 SibSp, Parch는 상관도가 높은 듯 하다.
    # 만약 Dead(0)면 생존율이 제일 낮은 Age값으로 ( => Old 70대 이상)
    # Survived(1)이면 생존율이 제일 높은 Age값으로 ( => Kids 10살 미만)

    for i in range(len(df_train)):
        if pd.isnull(df_train.loc[i, "Age"]):
            if df_train.loc[i, "Survived"] == 0: # Dead라면 70대 이상 값
                df_train.loc[i, "Age"] = 80
            else: # Survived(1) 이라면 Kids 10살 미만 -> 7살
                df_train.loc[i, "Age"] = 7

    for i in range(len(df_train)):
        age = int(df_train.loc[i, "Age"] / 10)
        df_train.loc[i,"Age"] = age

    # df_train["Age"] = df_train["Age"].map({0:"Kids", 1:"Teen", 2:"20s", 3:"30s",
    #                                        4:"40s", 5:"50s", 6:"60s", 7:"Old", 8:"Old"})
    # print(df_train["Age"].head(10))
    # show_group_rate("Age")

    # print(df_train.loc[5, "Age"]) # 결측치 들어왔는지 확인
    # print(df_train.info()) # Age 결측치 존재 확인 ==> 없음! :)

    # Embarked 확인
    df_train["Embarked"] = df_train["Embarked"].fillna("S")

    # 문자 데이터 숫자로 바꿔주기
    df_train["Sex"] = df_train["Sex"].map({"male" : 0, "female" : 1})
    df_train["Embarked"] = df_train["Embarked"].map({"Q":0, "C":1, "S":2})

    ########## Random Forest 학습 ##########
    clf = RandomForestClassifier(max_depth=3, n_estimators=200) # n_estimators 는 생성할 tree의 개수

    Y = df_train["Survived"]
    X = df_train.drop("Survived", axis=1)
    clf.fit(X,Y)
    # print(clf.score(X,Y))


    ######### 테스트를 해봅시다. ==> 학습할 때랑 똑같이 넣어줘야 한다.
    df_test = pd.read_csv("test.csv")
    pId = df_test["PassengerId"] # 정답 파일 만들 때 사용하기 위해 PassengerId를 따로 저장한다.
    print(df_test.info())

    # 학습이랑 똑같이 전처리
    df_test.drop(["PassengerId", "Fare", "Cabin", "Ticket", "Name"],axis=1,inplace=True)
    # age 전처리 ==> 평균으로 넣어줌
    df_test["Age"] = df_test["Age"].fillna(df_test["Age"].mean()) # fillna : 결측치 채워줌
    # Embarked 비어있는 2개의 값을 날리거나 채운다(제일 많은 걸로)
    df_test["Embarked"] = df_test["Embarked"].fillna("S") # Southampton에서 가장 많이 탔으므로 s로 채워줌
    # 문자 데이터를 숫자로 바꿔준다.
    df_test["Sex"] = df_test["Sex"].map({"male" : 0, "female" : 1})
    df_test["Embarked"] = df_test["Embarked"].map({"Q":0, "C":1, "S":2})

    # 분류기에 넣고 결과를 본다.
    result = clf.predict(df_test)
    # # 위 result를 데이터프레임에 넣어서 파일로 만들어 준다.
    # submit = pd.DataFrame({"PassengerId" : pId,
    #                        "Survived" : result})
    # submit.to_csv("submit.csv", index=False) # csv파일로 저장, 여기 index는 DF만들 때 자동으로 만들어주는 index ==> 표시하지 말라고


    # 우리가 학습한 결과와 groundtruth.csv(정답 파일)을 비교해보자
    gt = pd.read_csv("groundtruth.csv")

    hit = 0 # 맞으면 hit
    miss = 0 # 틀리면 miss

    for i in range(len(result)):
        if result[i] == gt.loc[i, "Survived"]:
            hit += 1
        else:
            miss += 1

    print(hit, miss, hit/(hit+miss)) # hit, miss, hit율

def ageVar2():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    # 필요없는 Column 제거
    df_train.drop(["PassengerId", "Cabin", "Name", "Ticket"], axis=1, inplace=True)
    # 결측값 존재하는 샘플 제거
    df_train = df_train.dropna()

    # Age 값이 소숫점을 갖는 자료 확인
    # outlier = df_train[df_train["Age"] - np.floor(df_train["Age"]) > 0]["Age"]
    # print(outlier)
    # print(len(outlier))
    # 이상치 처리 ==> 소수점 값 갖는 나이 처리
    df_train = df_train[df_train["Age"] - np.floor(df_train["Age"]) == 0]
    # 문자 데이터를 숫자로 바꿔준다.
    df_train["Sex"] = df_train["Sex"].map({"male" : 0, "female" : 1})
    df_train["Embarked"] = df_train["Embarked"].map({"Q":0, "C":1, "S":2})

    ####### Random forest 학습
    clf = RandomForestClassifier(max_depth=4, n_estimators=500) # n_estimators 는 생성할 tree의 개수
    # feautre 데이터와 label 데이터 분리
    # print(df_train.head(10))
    X = df_train.drop("Survived", axis=1)
    y = df_train["Survived"]
    clf.fit(X, y)

    ######### 테스트를 해봅시다. ==> 학습할 때랑 똑같이 넣어줘야 한다.
    df_test = pd.read_csv("test.csv")
    pId = df_test["PassengerId"] # 정답 파일 만들 때 사용하기 위해 PassengerId를 따로 저장한다.

    # 학습이랑 똑같이 전처리
    df_test.drop(["PassengerId", "Cabin", "Name", "Ticket"], axis=1, inplace=True)
    # 결측값 제거
    df_test = df_test.dropna()
    # Age 값이 소숫점을 갖는 자료 확인
    # outlier = df_test[df_test["Age"] - np.floor(df_test["Age"]) > 0]["Age"]
    # print(outlier)
    # print(len(outlier))
    # 이상치 처리 ==> 소수점 값 갖는 나이 처리
    df_train = df_train[df_train["Age"] - np.floor(df_train["Age"]) == 0]

    # 문자 데이터를 숫자로 바꿔준다.
    df_test["Sex"] = df_test["Sex"].map({"male" : 0, "female" : 1})
    df_test["Embarked"] = df_test["Embarked"].map({"Q":0, "C":1, "S":2})

    # 분류기에 넣고 결과를 본다.
    result= clf.predict(df_test)
    # # 위 result를 데이터프레임에 넣어서 파일로 만들어 준다.
    # submit = pd.DataFrame({"PassengerId" : pId,
    #                        "Survived" : result})
    # submit.to_csv("submit.csv", index=False) # csv파일로 저장, 여기 index는 DF만들 때 자동으로 만들어주는 index ==> 표시하지 말라고


    # 우리가 학습한 결과와 groundtruth.csv(정답 파일)을 비교해보자
    gt = pd.read_csv("groundtruth.csv")

    hit = 0 # 맞으면 hit
    miss = 0 # 틀리면 miss

    for i in range(len(result)):
        if result[i] == gt.loc[i, "Survived"]:
            hit += 1
        else:
            miss += 1

    print(hit, miss, hit/(hit+miss)) # hit, miss, hit율

ageVar()
ageVar2()