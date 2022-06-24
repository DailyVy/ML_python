import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    (train_set, train_label), (test_set, test_label) =\
        tf.keras.datasets.mnist.load_data("mnist.npz")

    clf = joblib.load("rf_mnist.pkl")
    # clf.score(train_set, train_label) # predict해서 결과를 보고 label과 비교
    # clf.score(test_set, test_label) # 에러나요 => shape이 달라서!!

    # train_set, test_set을 reshape한다
    train_set = train_set.reshape(60000, 784) # 784개의 feature가 있다고 생각하면 된다.
    test_set = test_set.reshape(10000, 784)

    print(clf.score(train_set, train_label)) # 0.9999833333333333
    print(clf.score(test_set, test_label)) # 0.9697

    # test_set, test_label 을 score해보니 0.9697 이 나오는데 => 303개가 틀림
    # 그럼 뭐가 틀렸고 어떻게 틀렸는지 확인을 해보자고
    print(clf.predict([test_set[0]])) # predict의 결과는 list로 나온다. 하나여도 리스트 두개여도 리스트
    print(clf.predict(test_set[0:3])) # predict의 결과는 list로 나온다. 하나여도 리스트 두개여도 리스트
    print(test_label[0])

    # 전체에 대해서 봅시다
    # for i in range(len(test_set)):
    #     result = clf.predict([test_set[i]])
    #     if result[0] != test_label[i]: # 예측값과 정답이 다르면
    #         print(f'Predict = {result[0]}, Ans = {test_label[i]}')
    #         img = test_set[i].reshape(28, 28)
    #         plt.imshow(img)
    #         plt.show()


    # MLP로 학습해보기
    (train_set, train_label), (test_set, test_label) = tf.keras.datasets.mnist.load_data("mnist.npz")

    train_set = train_set.reshape(60000, 784)
    test_set = test_set.reshape(10000, 784)

    print(train_label[0]) # 5

    # One-Hot-Encoding으로 바꿔주기
    train_label = tf.keras.utils.to_categorical(train_label, 10)
    print(train_label[0]) # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    test_label = tf.keras.utils.to_categorical(test_label, 10)
