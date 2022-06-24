import pandas as pd
from matplotlib import pyplot as plt

from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

#### pandas ####
# df = pd.read_csv("salmon_bass_data.csv")
#
# print(df.shape) # shape는 함수 아님! 이거 행,열을 튜플로 반환해 (318, 3)
# print(df.info()) # 피쳐들의 기본 정보 알 수 있음(결측치와 데이터 타입) ==> non-null
#
#
# ### 통계
# print(df.describe()) # 통계정보
#
# print(df.Length.mean()) # 길이 평균
# # print(df["Length"].mean()) # 위와 같음
#
# print(df["Lightness"].median()) # 밝기 중간값
# # print(df.Lightness.median()) # 위와 같음
#
# print(df["Length"].max())
# print(df.Length.min())
# print(df.Lightness.max())
# print(df["Lightness"].min())
#
# print(df.Length.std()) # 표준편차
# print(df["Lightness"].std())
#
# # 상관계수 : 두 열 사이의 상관계수, 1에 가까울수록 상관도가 높다
# print(df[["Length", "Lightness"]].corr(), "\n") # 두 열 사이의 상관계수
# print(df.corr()) # 모든 열 사이의 상관계수, 어차피 이 데이터는 컬럼 두개 뿐이라서..^^


##### Matplotlib ####
# 일단 먼저 ai_score_data.csv로
# df = pd.read_csv("ai_score_data.csv")

# 히스토그램으로 각 과목별 점수 분포 확인
# plt.hist(df["English"])
# plt.hist(df["Math"])
# plt.show()

# 이제 꾸며보자
# plt.title("Math Score")
# plt.xlabel("Score") # x축 label
# plt.xticks(range(0, df["Math"].max(), 5)) # 범위, step ==> 데코일 뿐
# plt.ylabel("Count")
# plt.hist(df["Math"], bins=20) # bins는 주머니라고 생각하면 돼 bins=20이면 20개의 주머니로 나누는 거야,
# # 데이터가 100개 있으면 5개씩 들어가겠지, default는 10인듯?
# plt.show()

# 산점도를 볼까요? ==> 두 범주 사이의 상관도
# plt.scatter(df["Math"], df["English"])
# plt.show()

# # 이제 산점도도 꾸며보자
# canvas = plt.figure(figsize=(8, 8)) # 이미지 크기 조절, 변수 명은 아무거나 상관 없구나
# plt.grid() # 격자 표현
# # plt.scatter(df["Math"],df["English"], color="red", marker="P")
# # plt.show()
#
# # 점 하나씩도 가능
# plt.scatter(df.loc[0,"Math"], df.loc[0,"English"], color="violet", marker="D")
# plt.scatter(df.loc[1,"Math"], df.loc[1,"English"], color="gold", marker="h")
# plt.scatter(df.loc[2,"Math"], df.loc[2,"English"], color="Green", marker="<")
# # plt.scatter(df.loc[3,"Math"], df.loc[3,"English"], color="Brown", marker="*")
# plt.show()

# 범주에 따라 산점도 표현을 다르게~
# canvas = plt.figure(figsize=(7, 7))
# plt.xlabel("Math Score")
# plt.ylabel("English Score")
# # 남(Blue)여(Red)
# for i in range(len(df["Sex"])):
#     if df.loc[i, "Sex"] == "M":
#         plt.scatter(df.loc[i, "Math"], df.loc[i, "English"], color="Blue")
#     else:
#         plt.scatter(df.loc[i, "Math"], df.loc[i, "English"], color="red")
# plt.show()



##### 이제 Salmon, Bass!!! #####
# df = pd.read_csv("salmon_bass_data.csv") # 일단 데이터 불러오고
#
# # 히스토그램 그리기
# # plt.hist(df["Length"]) # 길이에 대한 히스토그램 ==> Salmon, Bass 한번에 나옴. 보기 어려워
# # plt.show() # 그래프를 보기 위해!
#
# # 클래스를 기준으로 나눠보자
# salmon = df.loc[df["Class"]=="Salmon"]
# bass = df.loc[df["Class"]=="Bass"]
# print(salmon)

# 길이에 대한 히스토그램을 그립시다.
# plt.title("Length Histogram")
# plt.hist(salmon["Length"], alpha = 0.5, label="Salmon")
# plt.hist(bass["Length"], alpha= 0.5, label="Bass")
# plt.legend(loc="best") # 제일 좋은 위치에 알아서 범례표시
# plt.show()
#
# # 밝기에 대한 히스토그램
# plt.title("Lightness Histogram")
# plt.hist(salmon["Lightness"], alpha=0.5, label="Salmon")
# plt.hist(bass["Lightness"], alpha=0.5, label="Bass")
# plt.legend(loc="best")
# plt.show()

# 이제 두 특징 사이의 산점도를 그려볼까
# canvas = plt.figure(figsize=(8,8))
# plt.title("Scatter")
# plt.xlabel("Length")
# plt.ylabel("Lightness")
# plt.grid()
# plt.scatter(salmon["Length"], salmon["Lightness"], color="blue", label="Salmon")
# plt.scatter(bass["Length"], bass["Lightness"], color="red", label="Bass")
# plt.legend(loc="best")
# plt.show()

# X = [[0,0],[1,1]]
# Y = [0,1]
#
# # for i in range(len(X)):
# #     if Y[i] == 0:
# #         plt.scatter(X[i][0], X[i][1], color = "red")
# #     elif Y[i] == 1:
# #         plt.scatter(X[i][0], X[i][1], color = "blue")
#
# # plt.xlabel('Features[0]')
# # plt.ylabel('Features[1]')
# # plt.xticks(np.arange(-1,3,1))
# # plt.yticks(np.arange(-1,3,1))
# # plt.grid()
# # plt.show()
#
# dtree = tree.DecisionTreeClassifier()
# dtree = dtree.fit(X,Y)
#
# tree.plot_tree(dtree)
# plt.show()
#
# result = dtree.predict([[2,2],[1,1],[0,0],[0,1]])
# print(result)

### Salmon / Bass 분류 ###

df = pd.read_csv("salmon_bass_data.csv")
X = []
Y = []

for i in range(len(df)):
    fish = [df.loc[i, "Length"], df.loc[i, "Lightness"]]
    X.append(fish)
    Y.append(df.loc[i,"Class"])
# print(X)

# classifire는 DTree
# dtree = tree.DecisionTreeClassifier()
# dtree = dtree.fit(X,Y)
#
# plt.figure(figsize=(15,15))
# tree.plot_tree(dtree, fontsize=8, filled=True)
# # plt.show()
#
# result = dtree.predict([[11,4.2]])
# print(result)

# 가지치기
dtree = tree.DecisionTreeClassifier(max_depth=3) # 가지치기 깊이는 3level까지
dtree = dtree.fit(X,Y)

plt.figure(figsize=(15,15))
tree.plot_tree(dtree, fontsize=8, filled=True,
               class_names=["Bass","Salmon"], feature_names=["Length", "Lightness"]) # 클래스 네임 알파벳순으로 적어줌
plt.show()

result = dtree.predict([[11,4.2]])
print(result)