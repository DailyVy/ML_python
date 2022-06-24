import pandas as pd
from matplotlib import pyplot as plt
import seaborn
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print(df_train.info())

print(df_train.corr())  # 컬럼별 상관도 보여준다.

# 새로운 함수를 알려드리겠습니다. value_counts()
print(df_train["Survived"].value_counts())  # 0(Dead)이 549, 1(Survived) 이 342
print(df_train["Pclass"].value_counts())  # 1등석 216, 2등석 184, 3등석 491명


# visualize 해봅시다.
# 히스토그램 그릴 때 Dead, Survived 나눠서 그려줘야 해. 한 컬럼에서 나눠줘야 한다구!

# survive = df_train.loc[df_train["Survived"] == 1].copy()  # copy는 원본은 건드리지 않겠다...! 복제본 만들기
# dead = df_train.loc[df_train["Survived"] == 0].copy()

# 히스토그램
# plt.hist(survive["Pclass"], alpha=0.5, label="Survived")
# plt.hist(dead["Pclass"], alpha=0.5, label="Dead")
# plt.legend(loc='best')
# plt.show() #  뭔가 마음에 안들어


# 히스토그램 마음에 안드는 관계 seaborn 라이브러리를 통해 x축으로 퍼지도록~
# 세상 좋은 countplot 기능이 있어요
seaborn.countplot(data=df_train, x="Pclass", hue="Survived") #  hue : 카테고리 기능을 한다.
plt.show()