import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf


def dfs(src, group_img, cur_x, cur_y, label):
    """
    :param src: 이미지
    :param group_img: 방문 유무 체크
    :param cur_x: starting point
    :param cur_y: starting point
    :param label: 칠하고 싶은 값 == 방문 표시할 숫자
    :return:
    """
    rows = src.shape[0]
    cols = src.shape[1]

    # 현재 위치를 스택에 넣고 시작
    stack = [[cur_x, cur_y]]

    # 8-방향 : dx, dy 가 0, 0 이면 cur_x, cur_y
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]

    # stack에 값이 있는 동안 실시
    while stack:
        # 스택에서 하나 꺼내서 방문 표시
        x, y = stack.pop()
        group_img[y, x] = label  # label 은 방문 표시할 숫자

        # 인접 픽셀에 방문 가능한 곳이 있다면 스택에 넣음
        for i in range(8):
            next_x = x + dx[i]
            next_y = y + dy[i]

            # 영역을 벗어나면 continue
            if next_x < 0 or next_x >= cols or next_y < 0 or next_y >= rows:
                continue

            # 이미지에 값이 있고 방문 전이면 스택에 push
            if src[next_y, next_x] != 0 and group_img[next_y, next_x] == 0:
                stack.append([next_x, next_y])


def imgProcessing(img, x, y, w, h):
    crop = img[y:y + h, x:x + w]  # 우리의 관심영역
    # resize
    resizeImg = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)  # 사이즈를 (28, 28)로 축소
    roi = resizeImg.reshape(1, 28, 28, 1)  # roi의 shape변경 train은 (60000, 28, 28, 1) 이었음
    return roi


src = "./img/handwritingNumber.jpg"

if __name__ == "__main__":
    img = cv2.imread(src)
    height, width = img.shape[:2]  # 803, 2408
    img = cv2.resize(img, (int(width * 0.5), int(height * 0.5)), interpolation=cv2.INTER_AREA)  # 크기를 1/4로 줄임

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, bin_img = cv2.threshold(gray, -1, 255, cv2.THRESH_OTSU)

    bin_img = 255 - bin_img  # 배경을 까맣게 하고(0), 글자를 희게 함(255) ==> 우리 학습 데이터가 이래

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.dilate(bin_img, k)

    bin_img = dst

    # 블롭 구하기
    n_blob, label_img, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

    # 블롭을 화면에 show
    show_img = img.copy()  # 원본 이미지에 표시할 거라서

    # n_blob : 0은 배경이므로 뺀다. 레이블 개수만큼
    for i in range(1, n_blob):
        x, y, w, h, area = stats[i]
        # 너무 작은 블롭은 제외
        if area > 20:
            if h / w > 4:  # height/width 비가 4가 넘으면
                x = x - (3 * w)
                w *= 6
            # 박스를 조금 더 크게 그리자
            x -= 10
            y -= 10
            w += 20
            h += 20
            cv2.rectangle(show_img, (x, y, w, h), (237, 189, 0), thickness=2)

            roi = imgProcessing(bin_img, x, y, w, h)

            # model에 넣자
            model = load_model("./model/MNIST_CNN.hdf5")
            predict = model.predict(roi)  # 확률로 나오겠지 (softmax)
            predictVal = predict.argmax()  # 예측값은 argmax()로 (어차피 인덱스 == 값 이라서)

            # 이제 글자를 넣읍시다.
            cv2.putText(show_img, f"{predictVal}", (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (175, 97, 232))

    cv2.imshow("numbers_blob", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()