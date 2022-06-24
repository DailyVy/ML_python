import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
import pandas as pd

# iris = load_iris()
# X, y = iris.data, iris.target
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y)
#
# tree.plot_tree(clf)
# plt.show()


## kaggle 에서 받은 csv 파일을 이용해봅시다
df = pd.read_csv("Iris.csv")
print(df)