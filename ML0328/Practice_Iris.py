import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
# from sklearn.datasets import load_iris

# sklearn에 있는 iris 데이터셋을 사용해보자
# iris = load_iris()
# X, y = iris.data, iris.target
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X,y)
#
# print(X)
# print(y)
#
# plt.figure(figsize=(15,15))
# tree.plot_tree(clf, fontsize=10,filled=True)
# plt.show()


#### Iris를 Decision Tree로 분류해보자
df = pd.read_csv("Iris.csv")
# print(df)

df.set_index("Id", inplace=True) # Id 를 인덱스로 바꿔주었다.
# print(df)

X = []
Y = []

for i in range(len(df)):
    X.append([df.iloc[i,0], df.iloc[i,1], df.iloc[i,2], df.iloc[i,3]])
    Y.append(df.iloc[i,4])

# print(X)
# print(Y)

dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X,Y)

# plt.figure(figsize=(15,15))
# tree.plot_tree(dtree, filled=True, fontsize=10,
#                class_names=["Iris-setosa","Iris-Versicolor", "Iris-Virginica"],
#                feature_names=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])
# # class_names과 feature_names를 알파벳 순으로 써줘야 하나보다. 순서를 다르게 썼더니 결과값이 바뀐다.
# plt.show()

# 예측
print(dtree.predict([[5.7, 4.4, 1.5, 0.4]])) # ['Iris-setosa']
print(dtree.predict([[4.8, 3.1, 1.6, 0.2]])) # ['Iris-setosa']
print(dtree.predict([[5.0, 2.0, 3.5, 1.0]])) # ['Iris-versicolor']
print(dtree.predict([[5.5, 2.6, 4.4, 1.2]])) # ['Iris-versicolor']
print(dtree.predict([[6.5, 3.0, 5.8, 2.2]])) # ['Iris-virginica']
print(dtree.predict([[4.8, 3.0, 1.4, 0.3]])) # ['Iris-setosa']
print(dtree.predict([[6.6, 3.0, 4.4, 1.4]])) # ['Iris-versicolor']
print(dtree.predict([[6.0, 2.2, 5.0, 1.5]])) # ['Iris-virginica']
print(dtree.predict([[6.1, 2.6, 5.6, 1.4]])) # ['Iris-virginica']
print(dtree.predict([[5.9, 3.0, 5.1, 1.8]])) # ['Iris-virginica']
# 실제 데이터 넣어서 확인
print(dtree.predict([[5.1,3.5,1.4,0.2]])) # 실제 Iris-setosa
print(dtree.predict([[6.7,3.1,4.4,1.4]])) # 실제 Iris-versicolor
print(dtree.predict([[6.5,3.2,5.1,2.0]])) # 실제 Iris-virginica

###









