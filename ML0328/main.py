import pandas as pd
# import matplotlib.pyplot as plt # 아래와 같은 말임!
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("ai_score_data.csv")
    # print(df)

    # print(df.shape) # shape 는 변수, (행, 열) 반환
    # print(df.info()) # info()는 함수, 요약, non-null 체크

    # 통계
    # print(df.describe()) # 통계 정보

    # print(df['Math'].mean()) # 평균
    # print(df.Math.mean())
    # print(df['Math'].median()) # 중간값
    # print(df.English.median())
    #
    # print(df.corr()) # 전체 상관도, 1에 가까울수록 높다

    #### 히스토그램 만들기 ####
    # plt.hist(df["Math"])
    # plt.show()

    # 옵션을 줘 봅시다
    # plt.title("Math Score")
    # plt.xlabel("Score")
    # plt.ylabel("Count")
    # plt.xticks(range(0, 100, 5)) # 이건 x축의 데코레이션일 뿐 5씩 나눈게 아니다.
    # plt.hist(df["Math"], bins=20) # bins 로 계급을 나눈다.
    # plt.show()

    ### Scatter plot 산점도 ###
    # plt.scatter(df["Math"], df["English"])
    # plt.show()

    # 그래프 꾸밉시다
    # canvas = plt.figure(figsize=(7.0, 7.0)) # 이미지 크기 조절
    # plt.grid() # 격자무늬 생김~
    # plt.yticks(range(0,100,10))

    # color, marker(Field Marker) 모양
    # plt.scatter(df["Math"], df["English"], color='red', marker='X')
    # plt.show()

    # 점 하나씩도 가능 df.loc[0, 'Math'] -> 0번인 사람의 수학 점수
    # plt.scatter(df.loc[0, 'Math'], df.loc[0, 'English'])

    # 범주에 따라 산점도 표현을 다르게! if ~ else
    canvas = plt.figure(figsize=(7.0, 7.0)) # 이미지 크기 조절
    plt.grid() # 격자무늬 생김~
    plt.xlabel("Math")
    plt.ylabel("English")

    for i in range(len(df)): # 23개까지
        if df.loc[i,'Sex'] == "M": # df.loc[행, 열], 남자일 때
            plt.scatter(df.loc[i,"Math"], df.loc[i,"English"], color="blue")
        else: # 여자일 때
            plt.scatter(df.loc[i,"Math"], df.loc[i,"English"], color="red")

    plt.show()