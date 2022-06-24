import cv2

if __name__ == '__main__':
    img = cv2.imread("Lenna.png") # 이미지 읽어오는 거
    # print(img.shape)
    # print(img[100][100])
    # 값을 바꿔보자
    # img[100][100] = [255, 255, 255] # 3채널이니까 3개 넣어줘야지

    # # 점을 여러개 -> 선을 그려보자
    # white = [255, 255, 255]
    # blue = [255, 0, 0]
    # green = [0, 255, 0]
    # red = [0, 0, 255]
    #
    # # for i in range(100, 500):
    # #     img[100][i] = white
    # # img[200, :] = green
    # # img[:, 100] = blue
    #
    #
    # # face = img[240:400, 217:375].copy()
    # #
    # # face[:, :, 2] = 255 # 0은 Blue, 1은 Green, 2는 Red ==> Blue를 Maximum
    # # cv2.imshow("Face", face)
    # #
    # # # 이미지 저장합시다.
    # # cv2.imwrite("face_red.jpg", face)
    #
    #
    # #
    # # b, g, r = cv2.split(img)
    # # cv2.imshow("Blue", b)
    # # cv2.imshow("Green", g)
    # # cv2.imshow("Red", r)
    # #
    # # # img = cv2.merge((r, g, b)) # r, g, b 순서로 하니 푸르딩딩... 무서웡...
    # # img = cv2.merge((b, g, r))
    #
    #
    #
    # ### 2교시
    # # 색을 바꿔봅시다.
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # RGB 채널의 평균을 구해서 넣어줌
    # cv2.imshow("gray", gray)
    #
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # 실제 포맷팅을 바꿔주는게 아니고 r채널을 h로 g채널을 s로 뭐 이런식으로 강제로 값을...
    # v = hsv[:, :, 2]
    # cv2.imshow("hsv", hsv)
    # cv2.imshow("hsv", v)
    # gray2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("gray2", gray2)
    # print(gray[100][100])
    # print(gray2[100][100])

    cv2.imshow("Lena", img) # 윈도우창 타이틀, 구조체
    cv2.waitKey(0) # 0은 무한대로 기다리는 것, 입력이 올때까지 이미지를 끄지 않고 기다림, 입력을 하나 주면 창이 닫힘
    # cv2.destroyAllWindows() # 사용한 모든 창을 다 닫아줘. 메모리를 다 돌려준다는 것

