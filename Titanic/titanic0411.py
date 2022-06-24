from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

def showCountPlot(feature):
    seaborn.countplot(data=df_train, x=feature, hue="Survived")
    plt.show()


def showPiePlot(feature):
    df_survive = df_train.loc[df_train["Survived"] == 1] # 생존자만 추출

    sur_info = df_survive[feature].value_counts(sort=False) # sorting하지말고 순서대로 나오도록
    # plt.pie(sur_info) # 이대로 하니 못 알아보니까 label을 달아주자
    plt.pie(sur_info, labels=sur_info.index, autopct="%0.1f %%") # 1 3 2 (좌석)을 인덱스로 표시하자, autopct를 통해 비율 숫자로 표시
    plt.show()

def showGroupRate(feature):
    df_survive = df_train.loc[df_train["Survived"] == 1] # 생존자만 추출
    df_dead = df_train.loc[df_train["Survived"] == 0] # 죽은자 추출

    sur_info = df_survive[feature].value_counts(sort=False)
    dead_info = df_dead[feature].value_counts(sort=False)

    # pieplot이 한번에 나오도록 subplot을 이용해봅시다~
    fig = plt.figure() # 그림판을 하나 받아왔다 생각해, 빈 종이 하나를~
    plt.title("Survival Rate of {0}".format(feature))

    # plt.pie([sur_info["male"], dead_info["male"]]) # male 생존자 비율
    # plt.show()
    # male뿐 아니라 다른 것도! 반복문을 이용해 해보자
    for i, index in enumerate(sur_info.index):
        fig.add_subplot(1, len(sur_info), i+1) # subplot의 (행,열,그림번호), 주의할 점: subplot에서 index는 1부터 시작한다. 따라서 i+1
        plt.pie([sur_info[index], dead_info[index]],labels=["Survived","Dead"],autopct="%0.1f %%")
        plt.title("Survival rate of {0}".format(index))

    plt.show()

df_train = pd.read_csv("train.csv")
# print(df_train.info())

# seaborn.countplot(data=df_train, x= "Pclass", hue="Survived")
# plt.show()
# seaborn.countplot(data=df_train, x="Sex", hue="Survived")
# plt.show()
# seaborn.countplot(data=df_train, x="Embarked", hue="Survived")
# plt.show()
# => 코드가 반복된다! showCountPlot 함수로 만듦!
# showCountPlot("Pclass")

# showPiePlot("Pclass")
# showPiePlot("Sex")
# showPiePlot("SibSp")

showGroupRate("Sex")
showGroupRate("Pclass")


# # 잠시 enumerate 함수를 해 봅시다!! ==> showGroupRate의 그래프를 한번에 출력하게 하기 위해서(subplot)
# letters = ["A","B","C","D"]
# # 인덱스와 요소 한번에
# for i in range(len(letters)):
#     print(i, letters[i])
# # 위와 같은 작동을 하게 해주는 함수가 바로 enumerate
# for i, name in enumerate(letters):
#     print(i,name)


## svm 사용해보기
# from sklearn.svm import SVC
# clf = SVC() # 이것만 바꿔주면 된다.
