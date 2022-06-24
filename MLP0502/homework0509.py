import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle
import seaborn as sn
import pandas as pd
import numpy as np

if __name__ == "__main__":
    (train_set, train_label), (test_set, test_label) = \
        tf.keras.datasets.mnist.load_data("mnist.npz")

    clf = joblib.load("rf_mnist.pkl")
    # clf.score(train_set, train_label) # predict해서 결과를 보고 label과 비교
    # clf.score(test_set, test_label) # 에러나요 => shape이 달라서!!

    # train_set, test_set을 reshape한다
    train_set = train_set.reshape(60000, 784)  # 784개의 feature가 있다고 생각하면 된다.
    test_set = test_set.reshape(10000, 784)

    print(clf.score(train_set, train_label))  # 0.9999833333333333
    print(clf.score(test_set, test_label))  # 0.9697

    # test_set, test_label 을 score해보니 0.9697 이 나오는데 => 303개가 틀림
    # 그럼 뭐가 틀렸고 어떻게 틀렸는지 확인을 해보자고
    print(clf.predict([test_set[0]]))  # predict의 결과는 list로 나온다. 하나여도 리스트 두개여도 리스트
    print(clf.predict(test_set[0:3]))  # predict의 결과는 list로 나온다. 하나여도 리스트 두개여도 리스트
    print(test_label[0])

    # 전체에 대해서 봅시다
    # for i in range(len(test_set)):
    #     result = clf.predict([test_set[i]])
    #     if result[0] != test_label[i]: # 예측값과 정답이 다르면
    #         print(f'Predict = {result[0]}, Ans = {test_label[i]}')
    #         img = test_set[i].reshape(28, 28)
    #         plt.imshow(img)
    #         plt.show()

    ######################### 과제 2022-05-09 #############################
    # Confusion Matrix를 그려봅시다.
    # 10x10크기의 2차원 배열을 만들어서 0으로 초기화 합니다.
    # confusion_matrix_list = [[0 for i in range(10)],
    #                          [0 for i in range(10)],
    #                          [0 for i in range(10)],
    #                          [0 for i in range(10)],
    #                          [0 for i in range(10)],
    #                          [0 for i in range(10)],
    #                          [0 for i in range(10)],
    #                          [0 for i in range(10)],
    #                          [0 for i in range(10)],
    #                          [0 for i in range(10)]]
    # print(confusion_matrix_list)

    # 정답 x에 대한 모델의 응답 y를 [x][y]에 저장합니다.
    # 정답이 0인데 0이라고 한 것을 [0][0]에 저장
    # 정답이 0인데 1이라고 한 것을 [0][1]에 저장
    # [정답][예측] 자리에 저장

    # # 더해지는지 test해봄
    # confusion_matrix_list[0][0] = confusion_matrix_list[0][0] + 1
    # confusion_matrix_list[0][0] = confusion_matrix_list[0][0] + 1
    # print(confusion_matrix_list)

    # # 2차원 배열에전부 넣어봅시다~~~
    # for i in range(len(test_set)):
    #     result = clf.predict([test_set[i]])
    #     confusion_matrix_list[test_label[i]][result[0]] += 1
    #
    # print(confusion_matrix_list)

    # # 근데 계속 for문 돌리면 시간 낭비야
    # # confusion_matrix_list를 pickle 형태로 저장
    # confusion_matrix = open("confusionMatrixList.pickle", "wb")
    # pickle.dump(confusion_matrix_list, confusion_matrix)
    # confusion_matrix.close()

    # pickle 파일 불러오기
    confusion_matrix = open("confusionMatrixList.pickle", "rb")
    confusion_matrix_list = pickle.load(confusion_matrix)
    print(confusion_matrix_list)

    print(np.array(confusion_matrix_list[0]) / sum(confusion_matrix_list[0]))

    # 이제 confusion matrix를 그려봅시다.
    df_cm = pd.DataFrame(confusion_matrix_list, index=[i for i in range(10)], columns=[i for i in range(10)])
    # 각 행의 합을 구해준 다음
    df_cm["sum"] = df_cm.sum(axis=0)
    print(df_cm)
    # 각 행을 합으로 나눠줄 것이다.
    df_cm = df_cm.iloc[:, :10] / df_cm["sum"]
    # df_cm = df_cm.iloc[:, :10] / df_cm["sum"] * 100
    print(df_cm.loc[0, 0])
    # Seaborn으로 그림
    print(df_cm)
    plt.figure(figsize=(15, 15))
    sn.heatmap(df_cm,
               annot=True,  # annot : 숫자 표시여부
               # fmt='d', # fmt='d'는 정수형태
               fmt=".2%",
               cmap="YlGnBu",  # cmap 컬러 형태
               linewidth=1,  # linewidth 는 line을 넣어줌, 선의 굵기 결정
               linecolor='white')
    plt.show()

    confusion_matrix.close()
