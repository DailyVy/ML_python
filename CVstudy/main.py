import cv2
import glob
import os

imgsrc = glob.glob("D:/ML_python/ComVision/*.png")
# print(imgsrc)
src = imgsrc[0]

if __name__ == "__main__":
    img = cv2.imread(src) # imread = image read
    print(img.shape) # (512, 512, 3) : row, col, RGB
    # print(img[100][100]) # [ 72  64 174] BGR => Blue가 72, Green이 64, Red가 174
    # img[100][100] = [255, 255, 255] # 값을 바꾸고 싶다면 3채널의 값을 넣어줘서 바꿔주면 됨 => 하얀색으로 바꿈

    white = [255, 255, 255]
    blue = [255, 0, 0]
    green = [0, 255, 0]
    red = [0, 0, 255]

    # for i in range(100, 500):
    #     img[100][i] = white
    # img[200, :] = green
    # img[:, 100] = blue

    # face = img[220:380, 217:375]
    face = img[220:380, 217:375].copy()

    # face[:, :, 0] = 255 # face의 0번(Blue) 채널을 255로 바꿔줌
    # face[:, :, 1] = 255 # face의 1번(Green) 채널을 255로 바꿔줌
    # face[:, :, 2] = 255 # face의 2번(Red) 채널을 255로 바꿔줌

    # 이미지 저장해보자
    # cv2.imwrite("/face_red.jpg", face) # 저장할 파일 경로 이름(문자열), 저장할 영상(NumPy배열)


    ## openCV의 split 함수 ==> 사용하지 말고 NumPy 인덱싱 이용하는 것이 좋다.
    # b, g, r = cv2.split(img)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    # cv2.imshow("Blue", b)
    # cv2.imshow("Green", g)
    # cv2.imshow("Red", r)

    ## cv2.merge
    # imgMerge = cv2.merge((r, g, b))
    # imgMerge = cv2.merge((b, g, r)) # 순서는 BGR 이다!
    # cv2.imshow("merge", imgMerge)


    ### cvtColor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray) # rgb 채널의 평균을 구함
    # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    # cv2.imshow("rgb", rgb)

    # BGR2HSV
    #  rgb채널과 hsv는 포맷이 다르다.
    #  세 채널값을 강제로 hsv값으로 따버린다. ==> 실제 포맷을 바꾸는 것이 아님
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv", hsv)

    # hsv에서 v채널만 분리해보자. 즉 밝기 채널
    # v = hsv[:, :, 2]
    # cv2.imshow("hsv-v", v) # 아까 r(img[:, :, 2])과 같다
    # 실제로는 gray 이미지랑 v랑 같아야 하는데, 정상적으로 converting을 안해서 그렇다.


    # gray 바꾼 걸 다시 color로 바꿀 수 있을까? ==> 불가
    # 하지만 채널은 3개다. (gray의 경우 채널 1개)
    gray_2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow("gray_2", gray_2)

    print(gray[100][100]) # 98
    print(gray_2[100][100]) # [98 98 98]

    # cv2.imshow("Face", face)
    # cv2.imshow("Lena", img)
    cv2.waitKey(0) # 이 상태에서 key 입력을 기다리는 것, 0이라면 무한대 시간만큼 기다림 ms 단위인가봐
    cv2.destroyAllWindows()

