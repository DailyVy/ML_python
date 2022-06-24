import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree as skTree

# df = pd.read_csv('salmon_bass_data.csv')

# 연어/농어가 한번에 나옴. 보기 어렵다.
# plt.hist(df["Length"]) # salmon, bass의 길이가 전부 다 나옴
# plt.show()

# salmon_df = df.loc[df["Class"]=="Salmon"] # Salmon 데이터프레임
# print(salmon_df)
# bass_df = df.loc[df["Class"]=="Bass"] # Bass 데이터프레임
# print(bass_df)

# Length에 대한 히스토그램을 그려보자
# plt.hist(salmon_df["Length"], alpha=0.5, label="Salmon")
# plt.hist(bass_df["Length"], alpha=0.5, label="Bass")
# plt.legend(loc='best') # location을 best로 해두면 알아서 제일 괜찮은 위치에 범례를 붙여준다.
# plt.show() # 그래프를 보면 어떤 생각이 들어야 하나?

# Lightness에 대한 히스토그램
# plt.hist(salmon_df["Lightness"], alpha=0.5, label="Salmon")
# plt.hist(bass_df["Lightness"], alpha=0.5, label="Bass")
# plt.legend(loc='best') # location을 best로 해두면 알아서 제일 괜찮은 위치에 범례를 붙여준다.
# plt.show() # 그래프를 보면 어떤 생각이 들어야 하나?


### Salmon / Bass 분류 - Decision Tree ###

df = pd.read_csv("salmon_bass_data.csv")
# print(df, len(df)) # 318

X = [] # 빈 리스트를 만들었음 [[2, 0.8],[2, 0.8],...] 형태로 넣어줘야 해
Y = [] # ["Salmon", "Salmon", ...., "Bass"] ==> 자동화 코드를 만들어보자

# 자동화 코드
for i in range(len(df)): # 데이터 끝까지 가야지
    fish = [df.loc[i, "Length"], df.loc[i, "Lightness"]]
    X.append(fish)
    Y.append(df.loc[i, "Class"])

# print(X)
# print(Y)

# 학습
# dt_model = skTree.DecisionTreeClassifier(max_depth=3) # 가지치기
dt_model = skTree.DecisionTreeClassifier()
dt_model = dt_model.fit(X, Y)

# 어떻게 학습됐는지 보고싶다.
# plt.figure(figsize=(10,7))
# skTree.plot_tree(dt_model, fontsize=8, filled=True,
#                  class_names=["Salmon","Bass"],
#                  feature_names=["Length", "Lightness"]) # filled 트리안에 색 채우기
# plt.show()

# classifier는 DTree


## 220329 - 인공지능 개론시간에 갑자기 ML...
# 학습한 모델 사용해보기
result = dt_model.predict([[26, 1.2], [3, 4.0]]) # ['Salmon', 'Bass']
print(result)
