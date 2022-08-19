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
        group_img[y, x] = label # label 은 방문 표시할 숫자

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
    # print(f"resizeImg.shape : {resizeImg.shape}")
    # roi = np.expand_dims(resizeImg, axis=-1) # 채널 추가
    # roi = np.expand_dims(roi, axis=0) # 배열 차원 추가 (우리의 train set은 60000, 28, 28, 1)
    roi = resizeImg.reshape(1, 28, 28, 1)  # roi의 shape변경 train은 (60000, 28, 28, 1) 이었음
    return roi

src = "./img/handwritingNumber.jpg"

if __name__ == "__main__":
    img = cv2.imread(src)
    # print(img.shape) # 803, 2408, 3

    # 이미지 사이즈가 너무 크다.. resize 할래
    height, width = img.shape[:2] # 803, 2408
    img = cv2.resize(img, (int(width*0.5), int(height*0.5)), interpolation=cv2.INTER_AREA) # 크기를 1/4로 줄임
    """
    cv2.resize(src, dsize, fx, fy, interpolation)
    - 옵션
    src : 입력 영상, NumPy 배열
    dsize : 출력 영상 크기(확대/축소 목표 크기), 생략하면 fx, fy를 적용
        (width, height)
    fx, fy : 크기 배율, 생략하면 dsize를 적용
    interpolation : 보간법 알고리즘 선택
        일반적으로 축소에는 cv2.INTER_AREA, 확대에는 cv2.INTER_CUBIC, cv2.INTER_LINEAR
    """
    # print(img.shape) # 401, 1204, 3

    # cv2.imshow("numbers", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray_number", gray)

    th, bin_img = cv2.threshold(gray, -1, 255, cv2.THRESH_OTSU)
    # print(th) # 138.0
    # _, bin_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_TRUNC)
    # _, bin_img = cv2.threshold(bin_img, 250, 255, cv2.THRESH_TOZERO)
    # _, bin_img = cv2.threshold(gray, 150, 255, cv2.THRESH_TOZERO_INV)
    """
    cv2.threshold(img, 임계 값, value, ~) : 임계 값 이상의 값들을 value로 바꿔 준다.
    - 옵션
    cv2.THRESH_OTSU (오츠의 알고리즘) : 임계값(경계값)은 입력되도 무시된다.
    cv2.THRESH_BINARY : 픽셀 값이 임계 값을 넘으면 value, 넘지 못하면 0
    cv2.THRESH_BINARY_INV : 픽셀 값이 임계 값을 넘으면 0, 넘지 못하면 value
    cv2.THRESH_TRUNC : 픽셀 값이 임계 값을 넘으면 value, 넘지 못하면 원래의 값
    cv2.THRESH_TOZERO : 픽셀 값이 임계 값을 넘으면 원래의 값, 넘지 못하면 0
    cv2.THRESH_TOZERO_INV : 픽셀 값이 임계 값을 넘으면 0, 넘지 못하면 원래의 값
    """
    bin_img = 255 - bin_img # 배경을 까맣게 하고(0), 글자를 희게 함(255) ==> 우리 학습 데이터가 이래
    # cv2.imshow("bin_img", bin_img)

    ############################################################## 추가 영상처리 : 팽창 연산
    ##### morphology(형태학) : 노이즈 제거, 구멍 메꾸기, 연결되지 않은 경계 이어붙이기 등의 영상 연산
    # 숫자 선이 너무 얇아서 같은 숫자인데도 다른 영역이라고 인식된다.
    # 그래서 연결되지 않은 경계를 이어붙이기 위해 영상 연산을 진행한다. ==> 팽창 연산

    # 구조화 요소 커널 생성, 사각형 (3x3) 생성
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    """
    cv2.getStructuringElement(shape, ksize[, anchor]) : 구조화 요소 커널 생성 함수
    - 옵션
    - shape : 구조화 요소 커널의 모양
        cv2.MORPH_RECT : 사각형
        cv2.MORPH_ELLIPSE : 타원형
        cv2.MORPH_CROSS : 십자형
    - ksize : 커널 크기
    - anchor : 구조화 요소의 기준점, cv2.MORPH_CROSS에만 의미 있고 기본 값은 중심 점(-1, -1)
    """
    dst = cv2.dilate(bin_img, k)
    """
    cv2.dilate(src, kernel[, dst, anchor, iterations, borderType, borderValue])
    - 옵션
    - src : 입력 영상, NumPy 객체, 바이너리 영상(검은색 : 배경, 흰색 : 전경)
    - kernel : 구조화 요소 커널 객체
    - anchor : cv2.getStructuringElement와 동일, 기준점
    - iterations : 침식 연산 적용 반복 횟수
    - borderType : 외곽 영역 보정 방법 설정 플래그
    - borderValue : 외곽 영역 보정 값
    
    ==> 구조화 요소 커널을 입력 영상에 적용해서 1로 채워진 영역이 온전히 덮이지 않으면 1로 채워 넣는다.
    """

    # cv2.imshow("bin_img", bin_img)
    # cv2.imshow("dst", dst)

    bin_img = dst

    ######################################################################## 1. Flood-fill 구현
    # # Flood-fill (or Seed-fill)
    # # 다차원 배열의 어떤 칸과 연결된 영역을 찾는 알고리즘
    #
    # # 이진 이미지 그루핑
    # rows = img.shape[0] # 401
    # cols = img.shape[1] # 1204
    # group_img = np.zeros((rows, cols), dtype=np.uint8) # uint8 : 부호 없는 8비트 정수형
    #
    # label = 0
    #
    # for y in range(rows):
    #     for x in range(cols):
    #         if bin_img[y, x] != 0 and group_img[y, x] == 0: # bin_img 의 값이 0이 아니고 즉, 숫자 부분!
    #             label += 1
    #             dfs(bin_img, group_img, x, y, label)
    #
    # # group에 따라 다른 컬러로 show
    # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
    # group_img = cv2.cvtColor(group_img, cv2.COLOR_GRAY2BGR) # gray => color 하면 3채널의 값이 동일하게 들어가 있음
    #
    # # print(group_img[y, x, 0])
    #
    # for y in range(rows):
    #     for x in range(cols):
    #         # 3채널 값이 동일하므로 하나만 비교하면 됨
    #         if group_img[y, x, 0] != 0: # label 값으로 되어 있겠지
    #             color_idx = group_img[y, x, 0] % len(colors)
    #             group_img[y, x] = colors[color_idx]
    #
    # cv2.imshow("group", group_img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #################################################################### 2. OpenCV 의 floodfill
    # rows, cols = bin_img.shape[:2] # gray.shape
    # group_img = bin_img.copy()
    #
    # label = 50
    #
    # for y in range(rows):
    #     for x in range(cols):
    #         if group_img[y, x] == 255:
    #             label += 10
    #             cv2.floodFill(group_img, None, (x, y), label)
    #
    # cv2.imshow("group", group_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ######################################################### 3. OpenCV 의 connectedComponent
    # rows, cols = bin_img.shape[:2] # gray.shape 로 해도 됨
    # group_img = bin_img.copy()
    #
    # n_group, label_img = cv2.connectedComponents(bin_img, labels=None, connectivity=8, ltype=cv2.CV_16U)
    # print(label_img)
    """
    retval, labels = cv2.connectedComponents(src[, labels, connectivity=8, ltype])
    - 옵션
    - src : 입력 영상, 바이너리 스케일 이미지
    - labels : 레이블링 된 입력 영상과 같은 크기의 배열
    - connectivity : 연결성을 검사할 방향 개수(4, 8 중 선택) 
    - ltype : 결과 레이블 배열 dtype. 
        CV_8U : 8-bit unsigned integer: uchar ( 0..255 )
        CV_8S : 8-bit signed integer: schar ( -128..127 )
        CV_16U : 16-bit unsigned integer: ushort ( 0..65535 )
        CV_16S : 16-bit signed integer: short ( -32768..32767 )
        CV_32S : 32-bit signed integer: int ( -2147483648..2147483647 )
        CV_32F : 32-bit floating-point number: float ( -FLT_MAX..FLT_MAX, INF, NAN )
        CV_64F : 64-bit floating-point number: double ( -DBL_MAX..DBL_MAX, INF, NAN )
    - retval = 레이블 개수
    """
    #
    # #그룹에 따라 다른 컬러로 show
    # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
    # group_img = cv2.cvtColor(group_img, cv2.COLOR_GRAY2BGR)
    # # print(group_img[0, 0]) # [0 0 0]
    #
    # for y in range(rows):
    #     for x in range(cols):
    #         # 3 채널 값 동일 하므로 하나만 비교하면 된다.
    #         if label_img[y, x] != 0:
    #             color_idx = label_img[y, x] % len(colors)
    #             group_img[y, x] = colors[color_idx]
    #
    # cv2.imshow("group", group_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ######################################################### 4. OpenCV 의 connectedComponentWithStats

    # 블롭 구하기
    # n_blob, label_img, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
    """
    retval, labels, stats, centorids = cv2.connectedComponentsWithStats(src[, labels, stats, centroids, connectivity, ltype])
    - 옵션
    - src : 입력 영상, 바이너리 스케일 이미지
    - labels : 레이블링 된 입력 영상과 같은 크기의 배열
    - stats : N x 5 행렬 (N:레이블 개수)
    - centroids : 각 레이블의 중심점 좌표, N x 2 행렬 (N:레이블개수)
    - connectivity : 연결성을 검사할 방향 개수(4, 8 중 선택) 
    - ltype : 결과 레이블 배열 dtype. 
    - retval = 레이블 개수
    """
    # print(n_blob)
    # print(label_img)
    # print(stats)
    # print(centroids)

    # 블롭을 화면에 show
    # show_img = img.copy() # 원본 이미지에 표시할 거라서
    #
    # # n_blob : 0은 배경이므로 뺀다. 레이블 개수만큼
    # for i in range(1, n_blob):
    #     x, y, w, h, area = stats[i]
    #     # 너무 작은 블롭은 제외
    #     if h / w > 4: # height/width 비가 4가 넘으면
    #         x = x - (3*w)
    #         w *= 6
    #     if area > 20:
    #         cv2.rectangle(show_img, (x-10, y-10, w+20, h+20), (255, 0, 255), thickness=2)
    #
    # cv2.imshow("numbers_blob", show_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ######################################################### 5. 숫자 인식 프로그램

    # 블롭 구하기
    n_blob, label_img, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

    # 블롭을 화면에 show
    show_img = img.copy() # 원본 이미지에 표시할 거라서

    # n_blob : 0은 배경이므로 뺀다. 레이블 개수만큼
    for i in range(1, n_blob):
        x, y, w, h, area = stats[i]
        # 너무 작은 블롭은 제외
        if area > 20:
            if h / w > 4: # height/width 비가 4가 넘으면
                x = x - (3*w)
                w *= 6
            # 박스를 조금 더 크게 그리자
            x -= 10
            y -= 10
            w += 20
            h += 20
            cv2.rectangle(show_img, (x, y, w, h), (237, 189, 0), thickness=2)
            ################################################################## 여기까지가 blob

            ############################################ img processing
            roi = imgProcessing(bin_img, x, y, w, h)
            ############################################ end of img processing

            # model에 넣자
            model = load_model("./model/MNIST_CNN.hdf5")
            predict = model.predict(roi) # 확률로 나오겠지 (softmax)
            predictVal = predict.argmax() # 예측값은 argmax()로 (어차피 인덱스 == 값 이라서)

            # 이제 글자를 넣읍시다.
            cv2.putText(show_img, f"{predictVal}", (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (175, 97, 232))
            """
            cv2.putText(img, text, point, fontFace, fontSize, color [, thickness, lineType])
            - img : 글씨를 표시할 이미지
            - point : 글씨를 표현할 좌표
            - fontFace: 글꼴 => cv2.FONT_HERSHEY_PLAIN 뭐 이런거...
            """

    cv2.imshow("numbers_blob", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()