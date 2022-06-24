import numpy as np
import pandas as pd
import re # 정규 표현식을 지원하는 모듈
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

pd.set_option("display.max_columns", None) # 열 전체 출력

# print(train.head())
print(train.info())
print(test.info())

##################################### 데이터 전처리 ###############################################
# PassengerId, Survived, Pclass, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# PassengerId, Survived, Pclass 등의 컬럼은 int
# Todo : Name, Sex는 object이므로 int타입으로 바꿔주기
# 특히 Cabin 컬럼에서 null값이 많다. 전처리로 보정 필요

# List comprehension과 unique()함수를 사용하여 Cabin 컬럼의 고유값을 리스트로 살펴봅시다.
# pandas의 unique() : 유일한 값, 데이터의 고유값들이 어떠한 종류가 있는지 알고 싶을 때 사용
# print([(cab, type(cab)) for cab in train["Cabin"].unique()])
# ==> 값이 있는 컬럼은 string(eg. C85, E46, ...) null값은 float

# 원본 데이터 유지를 위한 복사본 생성
original_train = train.copy()

# train, test 합친 full data 생성
full_data = [train, test]
# full_data는 리스트이나 그 요소 하나하나는 데이터프레임으로 들어간다.

# Cabin컬럼 null값 보정
# x의 타입이 float 즉, null이면 0, else(String)이면 1 부여 -> train["HAS_Cabin"] 에 저장
# pandas의 apply() : 내가 정의해놓은 함수(여기서는 lambda x~)에 따라 전체 DF 혹은 특정한 column의 값을 일괄적으로 변경
# apply(lambda)의 형태로 많이 사용한다. lambda 입력변수: 리턴값
train["HAS_Cabin"] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test["HAS_Cabin"] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# SibSp and Parch을 결합해 FamilySize feature 새로 생성
# + 1 은 자신을 포함
for dataset in full_data:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
    # print(dataset["FamilySize"])

# FamilySize로 IsAlone Feature 새로 생성 ==> 즉 혼자일 경우
for dataset in full_data:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"]==1, "IsAlone"] = 1 # dataset["FamilySize"]==1 인 행 인덱스, "IsAlone" 컬럼 값은 1로 부여
    # print(dataset["IsAlone"])

# Embarked column 에서 null 값 제거 (fillna 합수)
for dataset in full_data:
    dataset["Embarked"] = dataset["Embarked"].fillna("S") # null값은 "S"로 대체 -> 최빈값

# Fare column에서 null값 제거(fillna 함수)
# qna : 왜 중앙값으로?
for dataset in full_data:
    dataset["Fare"] = dataset["Fare"].fillna(train["Fare"].median()) # null값 중앙값으로 대체

# Age column null값 제거
# qna : m-(n*σ) ~ m+(n*σ) => 정규분포의 구간, 현재 여기는 1시그마, 즉, 68.27 % 범위 포함
for dataset in full_data:
    age_avg = dataset["Age"].mean() # Age의 평균
    age_std = dataset["Age"].std() # Age의 표준편차
    age_null_count = dataset["Age"].isnull().sum() # isnull()은 NaN값 확인 가능, isnull().sum()은 결측치 개수 확인 가능
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count) # 결측치 개수 만큼 생성
    # 오류 방지
    dataset.loc[np.isnan(dataset["Age"]), "Age"] = age_null_random_list
    dataset["Age"] = dataset["Age"].astype(int) # float에서 int로 변환 => 나이는 정수니까

# passenger names로 title 컬럼 생성
# re.search() : 문자열 전체를 검색하여 정규식과 매치되는지 조사
# search()는 문자열의 처음부터가 아닌 문자열 전체를 검색한다.
# group() : match(), search() 메서드를 수행한 결과를 돌려주는 match객체의 메서드, 정규식 전체의 일치부를 찾는다.
def get_title(name):
    title_search = re.search("([A-Za-z]+)\.", name) # 정규표현식, name 중에서 매치 오브젝트를 리턴
    # if the title exists, extract and return it.
    if title_search:
        return title_search.group(1) # group(i)는 i번째 소괄호에 명시적으로 캡쳐된 부분만을 반환
    return ""

# print(get_title(train.loc[1, "Name"])) # Mrs

for dataset in full_data:
    dataset["Title"] = dataset["Name"].apply(get_title)

# 좀 독특한 title(eg. Lady, Countess, ...)은 Rare로 바꿔줌
# Title 의 고유값과 그 개수를 확인해보자~
# for dataset in full_data:
#     print(dataset["Title"].value_counts())

for dataset in full_data:
    dataset["Title"] = dataset["Title"].replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
                                                 "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
    dataset["Title"] = dataset["Title"].replace("Mlle", "Miss") # Mlle는 마드모아젤로 Miss와 의미가 같다
    dataset["Title"] = dataset["Title"].replace("Ms", "Miss") # Ms는 Miss, Mrs 둘 다 뜻이 있다.
    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs") # Mme은 Madame

# 고유값 가지수가 줄었는지 봅시다.
# for dataset in full_data:
#     print(dataset["Title"].value_counts())

# dataㅂ
for dataset in full_data:
    print(dataset["Age"].value_counts())
    print(dataset["Age"].describe())

for dataset in full_data:
    # Mapping Sex : 여자는 0, 남자는 1로
    print(dataset["Sex"].unique()) # 고유값들을 볼까요~
    dataset["Sex"] = dataset["Sex"].map({"female" : 0, "male" : 1}).astype(int)

    # Mapping titles : Mr는 1, Master는 2, Mrs는 3, Miss는 4, Rare은 5
    title_mapping = {"Mr":1, "Master":2, "Mrs":3, "Miss":4, "Rare":5}
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0) # 결측값은 0으로 채우기.. 없는거 같은데?

    # Mapping Embarked : S(사우스햄튼)항구는 0, C(쉘부르그)항구는 1, Q(퀸즈타운)항구는 2
    dataset["Embarked"] = dataset["Embarked"].map({"S":0, "C":1, "Q":2}).astype(int)

    # Mapping Fare
    # describe()로 4분위수 확인 : 25% - 7.91, 50% - 14.454, 75% - 31
    dataset.loc[dataset["Fare"]<=7.91, "Fare"] = 0
    dataset.loc[(dataset["Fare"]>7.91) & (dataset["Fare"]<=14.454) , "Fare"] = 1
    dataset.loc[(dataset["Fare"]>14.454) & (dataset["Fare"]<=31) , "Fare"] = 2
    dataset.loc[dataset["Fare"]>31, "Fare"] = 3
    dataset["Fare"] = dataset["Fare"].astype(int)

    # Mapping Age
    # 이건 그냥 16살씩 임의로 나눈듯?
    # Todo: 나중에 결과보고 Kids, 10대, 20대 이런식으로 나눠보자.
    dataset.loc[dataset["Age"] <= 16, "Age"] = 0
    dataset.loc[(dataset["Age"] > 16)&(dataset["Age"] <= 32), "Age"] = 1
    dataset.loc[(dataset["Age"] > 32)&(dataset["Age"] <= 48), "Age"] = 2
    dataset.loc[(dataset["Age"] > 48)&(dataset["Age"] <= 64), "Age"] = 3
    # dataset.loc[dataset["Age"] >= 64, "Age"]; # 이거 왜 세미콜론이지???? 난 4로 줄래
    dataset.loc[dataset["Age"] >= 64, "Age"] = 4


# 이제 불필요한 컬럼을 제거해주자, PassengerId, Name(Title로 이용), Ticket, Cabin(HAS_Cabin으로 대체), SibSp(FamilySize로 대체)
# qna : Parch는 왜 제거 안해주지???
drop_elements = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp"]
train = train.drop(drop_elements, axis=1) # 열에서 삭제
test = test.drop(drop_elements, axis=1)

print(train.head())
print(train.info()) # 모든 컬럼이 숫자형태로 바뀌었다.
print(test.info())


################################# 최적의 max depth찾기 ############################################
# 교차검증 : 훈련 데이터 셋을 바꿔가면서 훈련하여 나온 평균을 정확도로 보는 방법
# 한번의 학습을 통해 평가를 할 경우 과적합이 일어날 가능성이 크다. 이를 대비해 교차검증을 통해 과적합을 막아준다.
# K-fold 교차검증 : 학습 셋, 검증 셋을 나눠 반복해서 검증, k값만큼 폴드 셋에 k번의 학습과 검증, 이를 k번 평가

cv = KFold(n_splits=10) # 10개로 split하여, 10번의 학습과 검증을 할 것
accuracies = list() # 빈 리스트 생성
max_attributes = len(list(test)) # test는 10개의 컬럼을 가지고 있다.
depth_range = range(1, max_attributes + 1) # (1,11) 1부터 10까지

# 1부터 10(max_attributes)까지 max_depth 테스트 해보자
# for 문을 이용하여 여러 개의 decision tree를 만들고 각 tree별로 k-fold 교차 검증을 시행했을 때
# accuracy를 평균내서 가장 값이 높은 것으로 선택하면 된다.
for depth in depth_range:
    fold_accuracy = []
    tree_clf = DecisionTreeClassifier(max_depth=depth) # 1부터 10까지 들어가겠지, 나는 한 3정도 원하는뎅
    # print("Current max depth: ", depth, "\n")

    for train_fold, valid_fold in cv.split(train): # 학습데이터를 KFold로 10개로 나눠서 train_fold(9), valid_fold(1)에 넣는다.
        f_train = train.loc[train_fold] # Extract train data with cv indices(인덱스의 복수)
        f_valid = train.loc[valid_fold]

        model = tree_clf.fit(X=f_train.drop(["Survived"], axis=1), y= f_train["Survived"]) # train data를 모델에 fit시킨다.
        valid_acc = model.score(X=f_valid.drop(["Survived"], axis=1), y=f_valid["Survived"]) # fold validation data로 정확도 계산
        fold_accuracy.append(valid_acc) # 폴드 당 정확도

    avg = sum(fold_accuracy) / len(fold_accuracy) # 이건 평균 정확도... Average accuracy
    accuracies.append(avg)
    # print("Accuracy per fold: ", fold_accuracy, "\n")
    # print("Average accuracy: ", avg, "\n")

# 결과를 봅시다. 데이터프레임형태로~
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy" : accuracies})
# df = df[["Max Depth", "Average Accuracy"]] # qna : 이건 왜 해주는거지? 어차피 컬럼명은 똑같은데?
print(df.to_string(index=False)) # 열을 문자열로 바꾸어주는 to_string, index는 없이!
# max_depth가 3일때 average accuracy가 좋다.

################################ 학습을 하자 ######################################
y_train = train["Survived"]
x_train = train.drop(["Survived"], axis=1).values # 행들의 데이터 [[]] 2차원 형태
x_test = test.values

# qna : 왜 random_state는 42로 한걸까?, 42가 많네?
tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42) # 불순도 계산방법은 entropy로! 이 외에 gini도 있다.
tree_clf.fit(x_train, y_train)

y_pred = tree_clf.predict(x_test)

# score() : 트리 성능 평가, test 데이터 중 올바르게 분류한 데이터의 비율을 반환
acc_decision_tree = round(tree_clf.score(x_train, y_train) * 100, 2)
print(acc_decision_tree) # train_set의 정확도는 83.16


# 우리가 학습한 결과와 groundtruth.csv(정답 파일)을 비교해보자
gt = pd.read_csv("groundtruth.csv")

hit = 0 # 맞으면 hit
miss = 0 # 틀리면 miss

for i in range(len(y_pred)):
    if y_pred[i] == gt.loc[i, "Survived"]:
        hit += 1
    else:
        miss+=1

print(hit, miss, hit/(hit+miss)) # hit, miss, hit율 : 324 94 0.7751196172248804

# qna : 이상하다? 정답 파일과 비교했을 때 예측률이 77.5% 밖에 안나오는데?
#  저 83.16 % 는 Decision Tree의 정확도, Train set을 얼마만큼 정확히 분류했는가이다.

# qna : full_data=[train, test]로 데이터 전처리를 full_data를 통해 다 해주었다.
#  이 경우 원래 train에도 저장이 되는가? 객체라서??? ==> 와 진짜 바뀐다.... 놀랍다.... 아래 코드 참조
# a = [1, 2, 3]
# b = [4, 5, 6]
#
# temp = [a, b]
#
# for i in temp:
#     print(i)
#     i[0] = 9
#
# print(temp) # [[9, 2, 3], [9, 5, 6]]
# print(a, b) # 세상에 정말 원본이 바뀌잖아? 놀랍다. [9, 2, 3] [9, 5, 6]
