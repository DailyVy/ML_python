import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# def myPredict(x):
#     if x["Sex"] == "male":
#         return 0 # dead
#     else:
#         return 1 # survived


df_train = pd.read_csv("train.csv")

df_train.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1, inplace=True)

# age의 값이 비어있는 부분을 평균으로 채움
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean()) # fillna : 결측치 채워줌

# Embarked 비어있는 2개의 값을 날리거나 채운다(제일 많은 걸로)
df_train["Embarked"] = df_train["Embarked"].fillna("S") # Southampton에서 가장 많이 탔으므로 s로 채워줌

# print(df_train.info()) # 값이 다 채워진 걸 볼 수 있다.

# 데이터 전처리 : 문자 데이터를 숫자로!!
df_train["Sex"] = df_train["Sex"].map({"male" : 0, "female" : 1})
df_train["Embarked"] = df_train["Embarked"].map({"Q":0, "C":1, "S":2})
# map은 딕셔너리, 원래 이렇게하면 안됩니다. 0, 1 은 값이 있고 없고, 또는 크기가 있다고 판단하기 때문에
# 공평하게 하려면 column의 수를 늘려서 [1 0 0], [0 1 0] [0 0 1] -> 이게 one hot encoding

########## Random Forest 학습 ##########
clf = RandomForestClassifier(max_depth=3, n_estimators=200) # n_estimators 는 생성할 tree의 개수

Y = df_train["Survived"]
X = df_train.drop("Survived", axis=1)
clf.fit(X,Y)
# print(clf.score(X,Y))

######### 테스트를 해봅시다. ==> 학습할 때랑 똑같이 넣어줘야 한다.
df_test = pd.read_csv("test.csv")
pId = df_test["PassengerId"] # 정답 파일 만들 때 사용하기 위해 PassengerId를 따로 저장한다.

# 학습이랑 똑같이 전처리
df_test.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1, inplace=True)
# age의 값이 비어있는 부분을 평균으로 채움
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].mean()) # fillna : 결측치 채워줌
# Embarked 비어있는 2개의 값을 날리거나 채운다(제일 많은 걸로)
df_test["Embarked"] = df_test["Embarked"].fillna("S") # Southampton에서 가장 많이 탔으므로 s로 채워줌
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
        miss+=1

print(hit, miss, hit/(hit+miss)) # hit, miss, hit율

# myPredict 만들어보기
# # myPredict
# df_train = pd.read_csv("train.csv")
# df_test = pd.read_csv("test.csv")
# pId = df_test["PassengerId"]
#
# print(myPredict(df_test.loc[0])) # 첫번째 사람 dead ==> 남자인가봐
# print(myPredict(df_test.loc[1]))
#
# sur_sex = []
# for i in range(len(df_test)):
#     sur_sex.append(myPredict(df_test.loc[i]))
#
# submit_sex = pd.DataFrame({"PassengerId" : pId, "Survived" : sur_sex})
# submit_sex.to_csv("submit_sex.csv", index=False)
#
# gt = pd.read_csv("groundtruth.csv")
#
# hit = 0 # 맞으면 hit
# miss = 0 # 틀리면 miss
#
# for i in range(len(sur_sex)):
#     if sur_sex[i] == gt.loc[i, "Survived"]:
#         hit += 1
#     else:
#         miss+=1
#
# print(hit, miss, hit/(hit+miss)) # hit, miss, hit율