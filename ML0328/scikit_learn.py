from sklearn import tree
import numpy as np # arange 하려고 쓴거
from matplotlib import pyplot as plt

# 하드코딩으로 데이터 직접 만들기
# X = [[0, 0],[1, 1]] # [0,0] [1,1] 물고기 두마리~ [0, 0]은 Length, Lightness 임
# Y = [0, 1] # 클래스, ["Salmen", "Bass"] 도 가능, 지도학습을 하고 있으니 정답을 알아야 해
#
# for i in range(len(X)): # 2
#     if Y[i] == 0:
#         plt.scatter(X[i][0], X[i][1], color="red") # 데이터프레임이라면 X[i, 0] 이렇게 해야함
#     elif Y[i] == 1:
#         plt.scatter(X[i][0], X[i][1], color="blue")
#
# plt.xlabel('Feature[0]')
# plt.ylabel('Feature[1]')
# plt.xticks(range(-1, 2, 1))
# plt.yticks(range(-1, 2, 1))
# plt.grid()
# plt.show()

### Tree 만들기 ###
# X = [[0, 0],[1, 1]] # 특징들
# Y = [0, 1] # class, 정답
#
# dtree = tree.DecisionTreeClassifier()
# dtree.fit(X, Y)
# tree.plot_tree(dtree)
# plt.show()

