from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn import tree
from matplotlib import pyplot as plt

if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    # n_estimators : 생성할 tree의 갯수 ==> random forest의 tree가 조금씩 다른 데이터셋으로 만들어짐
    # max_dept : 최대 선택할 특성의 수
    # random_state : 기본적으로 bootstrap sampling(복원 추출)

    clf = clf.fit(X, y)

