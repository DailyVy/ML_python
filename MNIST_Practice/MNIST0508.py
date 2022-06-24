import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 데이터 셋을 로딩한다
# 자주 쓸 것이라서 오프라인에 다운로드 해두었다.
def load_dataset(online=False):
    if online:
        (tr_data, tr_label), (te_data, te_label) = tf.keras.datasets.mnist.load_data()
    else:
        path = "D:/ML_python/MNIST_Practice/mnist.npz"
        (tr_data, tr_label), (te_data, te_label) = tf.keras.datasets.mnist.load_data(path)
    return (tr_data, tr_label), (te_data, te_label)

def show_image(img, label): # img라는 이름의 2차원 배열을 받은 것
    # 이 img를 plt.imshow()의 인자로 넘겨줄 것임
    # plt.imshow(img) # 원하는 크기의 행렬을 만들어서 각 칸을 원하는 색으로 채우는 것
    plt.imshow(255-img, cmap="gray") # 255- 하면 색깔 반전, 그리고 grayscale로 가고싶다.
    print(label)
    plt.show()

def show_data_value(labels):
    count_value = np.bincount(labels)
    print(count_value)
    plt.bar(np.arange(0, 10), count_value)
    plt.xticks(np.arange(0, 10))
    plt.grid()
    plt.show()

def my_score(data, label):
    hit = 0
    miss = 0
    for i in range(len(data)):
        if clf.predict(data[i:i + 1]) == label[i:i + 1]:
            hit += 1
        else:
            miss += 1
    result = hit / (hit + miss)
    print(result)

def wrong_answer_note(data, label):
    wrong_answer = []
    showdata = data.reshape(len(data), 28, 28)
    inputVal = ""
    for i in range(len(data)):
        result = clf.predict(data[i:i+1])
        answer = label[i: i+1]
        if result != answer:
            print(f'모델 출력값은 {result}, 실제 정답은 {answer}')
            show_image(showdata[i], label[i])
            wrong_answer.append(result[0])
            inputVal = input("next? : ")
        if inputVal == "q":
            break
    print(f'오답노트 : {wrong_answer}')

if __name__ == "__main__":
    (train_set, train_label), (test_set, test_label) = load_dataset()
    # numpy에서 제공하는 행렬의 모양을 나타내주는 shape
    # print(train_set.shape) # (60000, 28, 28) 3차원배열, 28x28이 60000개
    # print(train_set[0].shape) # (28,28)
    # print(train_label.shape) # (60000,) # 60000개의 값이 한 행에 존재
    # print(train_label[:10]) # [5 0 4 1 9 2 1 3 1 4]
    # print(train_set[0]) # 첫번째 이미지 -> 5 겠지
    # print(train_set[0][10])
    # print(type(train_set[0][10][0]))
    # print(train_set[0][0][0])
    #
    # print(test_set.shape) # (10000, 28, 28)
    # print(test_label.shape) # (10000,)

    # 이미지를 봅시다. Visualize ==> matplotlib 이용
    # show_image라는 함수를 작성해봅시다. def show_image(img):
    # show_image(train_set[0])
    # show_image(train_set[1])
    # 이렇게 일일이 코드로 쳐서 이미지 띄우기 귀찮다 -> 코드를 작성하지
    # user로부터 입력을 받아서 이미지 띄우기 => 아무 입력이나 주면 다음 이미지로, q넣으면 quit
    userInput = "q"
    index = 0
    while userInput != "q":
        show_image(train_set[index], train_label[index])
        index += 1 # 다음거 봐야지
        userInput = input("next? : ")

    # 데이터의 분포를 확인해봅시다.
    # 정답을 보면 이미지가 몇 개 있는지 알 수 있다.
    print(type(train_label)) # <class 'numpy.ndarray'>

    # count_value = np.bincount(train_label) # pandas의 value_counts()랑 비슷
    # print(count_value) # [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949] ==> 0이 5923장, 1이 6742장...
    # plt.bar(np.arange(0, 10), count_value)
    # plt.xticks(np.arange(0, 10))
    # plt.grid()
    # plt.show()
    # # ==> show_data_value(labels)라는 함수를 만들었습니다.
    # show_data_value(train_label)
    # show_data_value(test_label)


    #### 학습을 합시다 ####
    # 학습을 해봅시당 RandomForest로~
    # clf = RandomForestClassifier()
    # clf.fit(train_set, train_label) # ValueError: Found array with dim 3. Estimator expected <= 2.

    # 에러가 발생했다.
    # train_set 이 3차원 배열이라서 그렇다
    # 28x28인 2차원 배열을 1차원(784)으로 바꿔주자, 그럼 전체적으로 2차원이 되겠지, (60000, 784)
    # reshape을 이용하면 된다.
    train_set = train_set.reshape(len(train_set), 784) # 마치 feautre가 784개 인것처럼
    print(train_set.shape) # (60000, 784)
    # clf = RandomForestClassifier()
    # clf.fit(train_set, train_label)

    # 실행할 때마다 학습하면 시간낭비~
    # 학습된 모델을 저장해두고 불러 씁시다.
    # sklearn 의 joblib이 있는데 dump와 load 메서드를 이용할 것이다.
    # dump 는 저장, load는 불러오기 => pickle 형식으로~
    # 파이썬의 객체 자체를 바이너리형태로 저장하는 것을 pickle이라고 합니다.
    # clf = joblib.dump(clf, "rf_mnist.pkl") # 모델, 모델이름

    # 학습한 모델 저장했으면 이제 fit은 필요 없어!! load만 하면됨^_^
    clf = joblib.load("rf_mnist.pkl") # 파일명을 인자로 준다.
    # 가져왔으니 테스트를 해봅시다
    # 가져온거라서 얘가 RandomForest인지 몰라... predict는 그냥 쳐줘...
    print(clf.predict(train_set[0:1])) # train_set[0]하면 1차원이라 안됨.. 2차원으로 만들어줘야해
    print(clf.predict(train_set[:10])) # 예측값
    print(train_label[:10]) # 실제값

    # Test Set에 대해서 몇 개 맞췄는지 보고싶다.
    # TestSet에 대하여 검증해봅시다.
    # RandomForest는 score라는 함수가 있다
    # score에 data set, label을 주면 비교해서 hit, miss계산해서 돌려준다.
    # test_set도 2차원으로 변경해주자
    test_set = test_set.reshape(len(test_set), 784) # (10000, 784)
    # 이제 score를 봅시다. 모델의 출력이랑 정답을 비교해서 score를 계산합니다.
    print(clf.score(test_set, test_label)) # 0.9699 나왔넹


    print(clf.predict(test_set[10:11]))
    print(test_label[10])

    # todo. score를 함수화 해보자. ==> my_score(data, label)
    # score 계산하기
    # hit = 0
    # miss = 0
    # for i in range(len(test_set)):
    #     if clf.predict(test_set[i:i+1]) == test_label[i:i+1]:
    #         hit += 1
    #     else:
    #         miss += 1
    # result = hit/(hit+miss)
    # print(result)

    # my_score 함수 사용하기
    # my_score(test_set, test_label) # 0.9699 로 score값과 같이 나온다~~

    # todo. 틀렸을 때 왜 틀렸는지 알고싶다. 요게 숙제! 내 모델은 요거랬는데 정답은 요거다 요게 숙제임댜
    # 오답노트
    # wrong_answer = []
    # for i in range(len(test_set)):
    #     result = clf.predict(test_set[i:i+1])
    #     answer = test_label[i: i+1]
    #     if result != answer:
    #         print(f'모델 출력값은 {result}, 실제 정답은 {answer}')
    #         wrong_answer.append(result)

    # 오답노트 함수로 만들기 : 이미지 팝업
    # wrong_answer_note()
    wrong_answer_note(test_set, test_label)