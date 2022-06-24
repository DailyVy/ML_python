
# 1. 직관적이고 조금 무식한 방법으로 x를 2차원으로 바꿔준 sin함수
def sinGraph():
    import math
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    x = []
    y = []

    # 일단 x에는 라디언값, y에는 sin값을 넣어보자
    for i in range(0, 361, 30):
        x.append(round(i*math.pi/180, 2))
        y.append(math.sin(round(i*math.pi/180, 2)))
    plt.scatter(x, y)
    # plt.show()

    # x는 1차원 배열이므로 lr에 넣을 수 없기 때문에 2차원으로 바꾼다.
    # 왜 우리 분류모델에서 fit시킬 때 x는 전부 2차원 형태였다.
    print(x) # 일단 x를 볼까
    x_2d = []
    for i in x:
        x_2d.append([i])

    # 이제 회귀모델에 넣어보자
    lr = LinearRegression()
    lr.fit(x_2d, y)

    plt.plot(x, lr.predict(x_2d)) # 그래프의 종속변수는 회귀모델!
    plt.show()

# sinGraph()

# 2. numpy 를 사용하여 x를 2차원으로 바꾼 sin 함수
def sinGraphNp():
    import math
    from matplotlib import pyplot as plt
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # np.linspace() : 숫자로 된 시퀀스 생성 (start, stop, breakpoint) stop이 마지막 값으로 포함된다.
    x = np.linspace(0, 2*math.pi, 12) # 0~2π까지 12개로 나눴으면 30도씩 들어가겠네~
    y = np.sin(x)
    plt.scatter(x, y)
    # plt.show()

    lr = LinearRegression()

    # np.reshape() : 배열의 차원을 변경할 때 사용 np.reshape(변경할 배열, 차원) or (차원)
    # reshape(-1, n) : 행의 자리에 -1이면, 변환될 배열의 행의 수는 알아서 지정
    # reshape(-1, 1) ==> 열은 1개고 행은 알아서 계산해서 배열을 변환 시켜줌
    lr.fit(x.reshape(-1, 1), y)
    plt.plot(x, lr.predict(x.reshape(-1, 1)))
    plt.show()

# sinGraphNp()


# 3. 다중회귀 : sin함수 다중회귀는 안될거지만 일단 해보자
def sinMulLr():
    import math
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures # feature를 다항식 feature로 변환해주는 클래스 이용
    import numpy as np

    x = np.linspace(0, 2*math.pi, 12)
    x = x.reshape(-1, 1) # x를 2차원 배열로~
    y = np.sin(x)
    plt.scatter(x, y)
    # plt.show()

    lr = LinearRegression()

    # polynomial feature를 이용하여 x의 차원을 올린다.
    # 다중회귀 하려면 이렇게 feature의 차원을 올려야 하나보다.
    poly_feat = PolynomialFeatures(degree=4) # 2차원으로
    x_poly = poly_feat.fit_transform(x)

    # 변화된 x에 대해서 선형회귀 한다.
    lr.fit(x_poly, y)

    plt.plot(x, lr.predict(x_poly))
    plt.show()

# sinMulLr()
# 궁금한 점 : 다항회귀랑 다중회귀의 차이는? 이 경우는 x의 차수를 올려주는 거니 다항회귀 아닌가?
# 그러니까 선형회귀모델이 다항식처럼 나오는 거인가?


# 2차원 데이터로 다중회귀? 다항회귀? 테스트~_~
def twoDLr():
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    n = 100

    # np.random.rand(m, n) : 0 ~ 1 균일분포 표준정규분포 난수를 matrix array(m, n) 생성
    x = 6 * np.random.rand(n, 1) - 3 # x는 -3 ~ 3 사이의 값
    y = 0.5 * x**2 + x + 2 + np.random.rand(n, 1)
    # Y = 0.5 * x**2 + x + 2
    # plt.scatter(x, Y, s=3, label='Y')
    plt.scatter(x, y, s=3, label='y')
    # plt.legend(loc='best')
    # plt.show()

    poly_feat = PolynomialFeatures(degree=2)
    x_poly = poly_feat.fit_transform(x)

    lr = LinearRegression()
    lr.fit(x_poly, y)
    plt.scatter(x, lr.predict(x_poly), s=3, label = 'predict')
    plt.legend(loc='best')
    plt.show()

# twoDLr()

# 아까 sin함수 테스트 데이터를 더 늘려보자
def sinMulLrMore():
    import math
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures # feature를 다항식 feature로 변환해주는 클래스 이용
    import numpy as np

    x = np.linspace(-2*math.pi, 4*math.pi, 40)
    x = x.reshape(-1, 1) # x를 2차원 배열로~
    y = np.sin(x)
    plt.scatter(x, y)
    # plt.show()

    lr = LinearRegression()

    # polynomial feature를 이용하여 x의 차원을 올린다.
    # 다중회귀 하려면 이렇게 feature의 차원을 올려야 하나보다.
    poly_feat = PolynomialFeatures(degree=13)
    x_poly = poly_feat.fit_transform(x)

    # 변화된 x에 대해서 선형회귀 한다.
    lr.fit(x_poly, y)

    plt.plot(x, lr.predict(x_poly), color='red')
    plt.show()

# sinMulLrMore()


# 인터넷에 찾아본 다중회귀 모델(https://www.w3schools.com/python/python_ml_multiple_regression.asp)
def mulLr():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from matplotlib import pyplot as plt

    df = pd.read_csv("cars.csv")

    X = df[['Weight', 'Volume']]
    y = df['CO2']

    regr = LinearRegression()
    regr.fit(X, y)

    # predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm^3
    predictCO2 = regr.predict([[2300, 1300]])

    print(predictCO2)

mulLr()

# Overfitting / Underfitting