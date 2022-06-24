import pandas as pd
import random as r

def score():
    return r.randint(0, 100) # randint 범위를 포함한 범위

# def add(a, b):
#     return a + b
#
# def sub(a, b):
#     return a - b
#
# def mul(a, b):
#     return a * b
#
# def div(a, b):
#     if b == 0:
#         return "0으로 나눌 수 없습니다."
#     else:
#         return a / b


if __name__ == '__main__': # 여기가 메인함수라는 걸 명시
    # df = pd.read_csv("salmon_bass_data.csv")
    # print(df)
    dic = {"name": "강현우"}
    print(dic, type(dic))
    print(dic["name"]) # 딕셔너리에서는 key(column) 이름 자체를 index 처럼 사용할 수 있다.
    dic["name"] = "김근형" # 새로운 값 할당가능
    print(dic)

    nameList = ["강현우", "김근형", "김동혁"]
    dic["name"] = nameList
    print(dic)

    dic["id"] = [0, 1] # 3명이지만 일부러 두 명만 만들어봄
    print(dic)
    dic["id"].append(2) # dic["id"] 자체가 하나의 list이므로 리스트연산 가능
    print(dic)

    df = pd.DataFrame(dic)
    print(df)

    df["국어"] = 0
    print(df)


    # df["국어"] = [100, 95, 90, 80] # 에러납니다~ ValueError: Length of values (4) does not match length of index (3)
    df["국어"] = [100, 95, 90]

    print(df)

    df["국어"] = [score(), score(), score()]
    df["영어"] = 0
    df["수학"] = 0
    print(df)

    # print("\n=============사칙연산 함수 만들기==============\n")
    # print(add("22", "32"))
    # print(sub(2, 3))
    # print(mul(3, 2))
    # print(div(3, 0))


# Dataframe에 행 접근!

    # loc[index]
    df.loc[0] = ["신주석", "2022-0", score(), score(),score()] # 0번 행에 통째로 값을 다 입력
    print(df)

    df.loc[3] = ["남현진", "2022-3", score(), score(), score()]
    print(df)

    df.loc[4] = 0
    print(df)

    print(df.shape) # (5, 3) : 5행 3열
    # print(df.shape[0]) # 행 5
    # print(df.shape[1]) # 열 3

    df.loc[df.shape[0]] = ["박수연", "2022-5", score(), score(), score()]
    print(df)

    # df.loc[1]["id"] = "2022-1" # 에러
    df.loc[1, "id"] = "2022-1"
    df.loc[2, "id"] = "2022-2"
    df.loc[4] = ["노영하", "2022-4", score(), score(), score()]
    print(df)
    print(len(df), df.shape[0]) # 6 6

    print(df)
    for i in range(len(df), 23): # df.shape[0], 23해도 됨
        df.loc[i] = ['name-{0}'.format(i), '2022-{0}'.format(i), score(), score(), score()]

    # df.loc[0, "name"]
    # df.loc[1, "name"]
    # df.loc[2, "name"]
    # df.loc[3, "name"]
    # df.loc[4, "name"]
    # df.loc[5, "name"]
    df.loc[6, "name"] = "심우석"
    df.loc[7, "name"] = "안원영"
    df.loc[8, "name"] = "오수은"
    df.loc[9, "name"] = "이근형"
    df.loc[10, "name"] = "이시형"
    df.loc[11, "name"] = "이은정"
    df.loc[12, "name"] = "이은주"
    df.loc[13, "name"] = "장민규"
    df.loc[14, "name"] = "전세환"
    df.loc[15, "name"] = "정경임"
    df.loc[16, "name"] = "차민욱"
    df.loc[17, "name"] = "최민석"
    df.loc[18, "name"] = "최비결"
    df.loc[19, "name"] = "최윤정"
    df.loc[20, "name"] = "최지호"
    df.loc[21, "name"] = "표주혁"
    df.loc[22, "name"] = "허진행"

    print(df)


# 영어, 수학 for문으로 값을 넣으니 float형 ==> int 전환 (이거 pandas 버그랭)
    # for i in range(0, 23):
    #     df.loc[i, "영어"] = score()
    #     df.loc[i, "수학"] = score()
    #
    # df = df.astype({"영어": "int"}) #
    # df = df.astype({"수학": "int"}) #
    #
    # print(df)
    # print(type(df.loc[3, "영어"]))

    df.to_csv("ai_engr.csv") # csv 파일로 저장

