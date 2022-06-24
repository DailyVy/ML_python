from matplotlib import pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
# numpy : 다차원 행렬을 처리하기 위한 라이브러리

# X = [0.0, 0.52, 1.05, 1.57, 2.09, 2.62, 3.14, 3.67, 4.19, 4.71, 5.24, 5.76]
# Y = [0.0, 0.5, 0.87, 1.0, 0.87, 0.5, 0.0, -0.5, -0.87, -1.0, -0.87, -0.5]

# plt.scatter(X, Y)
# plt.show()

# 사인 함수를 만들어보자
def sinGraph():
    X = []
    Y = []

    for i in range(0, 361, 30):
        X.append(i*math.pi/180)
        Y.append(math.sin(i*math.pi/180))

    # 사인함수 그래프 확인
    # plt.scatter(X, Y)
    # plt.show()

    lr = LinearRegression()

    # X를 2차원으로 만들어주자, 지금 X는 리스트니까
    # 1. 좀 무식하지만 직관적인 방법 (2번 방법은 numpy이용 아래 sinGraph2 참조)
    X_2d = []
    for i in X:
        X_2d.append([i])
    # print(X_2d)

    # 선형회귀 모델 만들자
    lr.fit(X_2d, Y)

    plt.scatter(X, Y) #
    plt.plot(X, lr.predict(X_2d)) # 만든 모델의 그래프를 보자~
    plt.show() # 결론은 사인함수는 선형이 안돼^^

# sinGraph()

# numpy를 활용해서 사인함수~
def sinGraph2():
    X = np.linspace(0, 2*math.pi, 12) # 0 ~ 2pi, 12구간
    Y = np.sin(X)

    plt.scatter(X, Y)
    # plt.show()

    # X를 reshape 해보자
    print(X)
    print(X.reshape(-1, 1)) # reshape 할 때 -1 하면 차원이 하나 늘어난다.

    X = X.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, Y)

    plt.plot(X, lr.predict(X))
    plt.show()


# sinGraph2()

# 2차원 데이터를 만들어서 다중선형회귀를 해보자
def twoDlr():
    n = 100

    x = 6 * np.random.rand(n, 1) - 3 # -3 ~ 3 까지의 값
    y = 0.5 * x**2 + x + 2 + np.random.rand(n, 1) # np.random.rand 로 Noise 추가
    # Y = 0.5 * x**2 + x + 2 # 이건 노이즈 안준 거

    # plt.scatter(x,Y, s=3, label='Y')
    plt.scatter(x, y, s=3, label='y')

    poly_feat = PolynomialFeatures(degree=2) # feature를 2차원으로 바꿔줌(다중 선형 회귀)
    x_poly = poly_feat.fit_transform(x) # 모델을 변경하는게 아니고 x의 차원을 바꿔줘야 하구나... 그냥 이렇게 사용을 해야 하는구나

    lr = LinearRegression()
    lr.fit(x_poly, y)

    plt.scatter(x, lr.predict(x_poly), s=3, label='lr') # s= 점의 사이즈 옵션
    plt.legend(loc='best')
    plt.show()

twoDlr()