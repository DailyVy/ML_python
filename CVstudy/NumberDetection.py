import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

src = "img/number.jpg"

def dfs(src, group_img, cur_x, cur_y, label):
    rows = src.shape[0]
    cols = src.shape[1]

    # 현재 위치를 스택에 넣고 시작
    stack = [[cur_x, cur_y]]

    # 8-방향
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]

    # stack에 값이 있는 동안 실시
    while stack:
        # 스택에서 하나 꺼내서 방문 표시
        x, y = stack.pop()
        group_img[y, x] = label # label은 방문표시할 숫자

        # 인접 픽셀에 방문 가능한 곳이 있다면 스택에 넣음
        for i in range(8):
            next_x = x + dx[i]
            next_y = y + dy[i]

            # 영역을 벗어나면 continue
            if next_x < 0 or next_x >= cols or next_y < 0 or next_y >= rows:
                continue

            # 이미지의 값이 있고 방문 전이면 스택에 push
            if src[next_y, next_x] != 0 and group_img[next_y, next_x] == 0:
                stack.append([next_x, next_y])


if __name__ == "__main__":
    img = cv2.imread(src)
    # print(img.shape) # (137, 536, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray_number", gray)

    # cv2.threshold(img, 임계값, value, ~)
    #  임계값 이상의 값들은 value로 바꿔준다.
    # th, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_OTSU)
    # print(th) # 206.0 이 나오는데 얜 뭐람 ==>OTSU알고리즘의 최적화된 thresholder 값인듯

    # th, bin_img = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    th, bin_img = cv2.threshold(gray, 181, 255, cv2.THRESH_BINARY_INV)
    # th, bin_img = cv2.threshold(gray, 250, 255, cv2.THRESH_TRUNC)
    # th, bin_img = cv2.threshold(gray, 181, 255, cv2.THRESH_TOZERO)
    # 즉, 여기서 0보다 큰 값들은 다 255로 즉 하얀색으로 바꿔주겠다는 뜻

    # cv2.THRESH_OTSU : 여기에서는 임계값을 설정해줘도 무시된다 => 알아서 최적의 임계값을 구함
    #  오츠의 알고리즘은 임계값을 임의로 정해서 픽셀을 두 부류로 나누고 두 부류의 명암분포를 구하는 작업을 반복
    #  모든 경우의 수 중에서 두 부류의 명암 분포가 가장 균일할 때의 임계값을 선택
    #  즉, 아까 print(th)에서 나온 206.0이 바로 그 임계값이네.
    # cv2.THRESH_BINARY : 픽셀 값이 임계값을 넘으면 value로 지정하고, 넘지못하면 0으로 지정
    # cv2.THRESH_BINARY_INV : cv2.THRESH_BINARY의 반대
    # cv2.THRESH_TRUNC : 픽셀값이 임계값을 넘으면 value로 지정, 넘지 못하면 0으로 지정
    # cv2.THRESH_TOZERO : 픽셀값이 임계값을 넘으면 원래값 유지, 넘지 못하면 0으로 지정

    # bin_img = 255 - bin_img # 만약 cv2.THRESH_BINARY를 해줬으면 이렇게 해줬을 건데 INV로 설정해서 필요없음
    # 배경을 까맣게 하고 글자를 희게 한 이유 ==> 우리가 학습을 그렇게 해서

    # cv2.imshow("bin", bin_img)

    # print(bin_img.shape)


    # 팽창 연산 cv2.dilate(src, kernel, [~])
    #  영상 속 사물의 주변을 덧붙여서 영역을 확장하는 연산

    # 구조화 요소 커널, 사각형 ( 3 x 3 ) 생성
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cv2.getStructuringElement(shape, ksize) ==> shape : 구조화 요소 커널의 모양 결정, ksize : 커널의 크기
    dst = cv2.dilate(bin_img, k)

    # cv2.imshow("dst", dst)



    ###########################################################################################
    # flood-fill 구현한거 (dfs)
    # rows = img.shape[0]
    # cols = img.shape[1]
    #
    # group_img = np.zeros((rows, cols), dtype=np.uint8) # uint8 : 0 ~ 255 의 양수만 표현 가능
    #
    # label = 0
    #
    # for y in range(rows):
    #     for x in range(cols):
    #         if bin_img[y, x] != 0 and group_img[y, x] == 0:
    #             label += 1
    #             dfs(dst, group_img, x, y, label)
    #
    # # group에 따라 다른 칼라로 show
    # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
    # group_img = cv2.cvtColor(group_img, cv2.COLOR_GRAY2BGR) # gray scale 을 3채널로 바꿈
    #
    # for y in range(rows):
    #     for x in range(cols):
    #         # 3채널 값이 동일하므로 하나만 비교하면 됨
    #         if group_img[y, x, 0] != 0:
    #             color_idx = group_img[y, x, 0] % len(colors)
    #             group_img[y, x] = colors[color_idx]
    #
    #
    # cv2.imshow("group", group_img)

    #############################################################################################
    # 그냥 opencv의 floodfill 사용 ==> 이게 dfs 부분이라고 생각해
    # rows, cols = gray.shape
    # group_img = dst.copy()
    #
    # label = 0
    #
    # for y in range(rows):
    #     for x in range(cols):
    #         if group_img[y, x] == 255:
    #             label += 1
    #             cv2.floodFill(group_img, None, (x, y), label)

    #############################################################################################
    # # 그냥 ConnectedComponent
    # rows, cols = dst.shape
    # # print(dst.shape)
    # group_img = dst.copy()
    #
    # n_group, label_img = cv2.connectedComponents(dst, labels=None, connectivity=8, ltype=cv2.CV_16U)
    # # ltyle : labels 행렬 타입
    # # print(n_group)
    # print(label_img)
    #
    # # group에 따라 다른 칼라로 show
    # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
    # group_img = cv2.cvtColor(group_img, cv2.COLOR_GRAY2BGR) # gray scale 을 3채널로 바꿈
    # # group_img = cv2.GaussianBlur(group_img, (5, 5), 0)
    # # print(group_img[0, 0])
    # # print(label_img)
    #
    # for y in range(rows):
    #     for x in range(cols):
    #         # 3채널 값이 동일하므로 하나만 비교하면 됨
    #         if label_img[y, x] != 0:
    #             color_idx = label_img[y, x] % len(colors)
    #             group_img[y, x] = colors[color_idx]
    #
    #
    # cv2.imshow("group", group_img)


    #############################################################################################
    # cv.connectedComponentsWithStats
    #  return 값이 retval, labels, stats, centroids
    """
    retval : blob의 갯수 (배경 포함, 배경은 0)
    labels : 레이블 된 이미지
    stats : 바운딩 박스. x, y, w, h, area
    centroids : 중점 좌표 
    """




    w_width = 40
    w_height = 80

    # for y in range(0, 100, 65):
    #     for x in range(536):
    #         crop = gray[y:y+w_height, x:x+w_width]
    #         # 이미지 resizing
    #         resizeImg = cv2.resize(crop, (28, 28))
    #         roi = np.expand_dims(resizeImg, axis=-1) # axis = -1 : 제일 뒷차원을 추가
    #         print(roi.shape)
    #         roi = np.expand_dims(roi, axis=0) # axis = 0 : 제일 앞차원
    #         # print(roi.shape)
    #         # roi = np.reshape(resizeImg, (1, 28, 28, 1))
    #
    #         model = load_model("./model/cnn_mlp_practice.h5")
    #         predict = model.predict(roi)
    #
    #         # print(predict)
    #         print(predict[0].argmax())
    #         print(predict[0])
    #
    #         # cv2.imshow("crop_img", crop)
    #         cv2.imshow("resize_img", resizeImg)
    #
    #         key = cv2.waitKey(33)
    #
    #         if key == 27:
    #             cv2.destroyAllWindows()
    #             exit(0)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

