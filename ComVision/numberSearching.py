import cv2
from tensorflow.keras.models import load_model # model 부르기 위해서
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("number.jpg") # 이미지 읽어오는 거
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    print(img.shape) # 137, 536, 3

    w_width = 50
    w_height = 70

    # y도, x도 100까지 움직이게 할거야
    for y in range(0, 100, 70):
        for x in range(500):
            crop = gray[y:y+w_height, x:x+w_width]
            # cv2.imshow("crop_img",crop)
            resizeImg = cv2.resize(crop, (28, 28))

            roi = np.reshape(resizeImg, (1, 28, 28, 1))
            print(roi)
            print(roi.shape)
            roi2 = np.expand_dims(resizeImg, axis=-1) # 제일 앞 차원을 추가
            roi2 = np.expand_dims(roi2, axis=0) #
            print("=========================")
            print(roi2)

            print(roi2.shape)

            cv2.imshow("resize_img", resizeImg)
            key = cv2.waitKey(33)

            model = load_model("./model/cnn_mlp.h5")
            predict = model.predict(roi)

            print(predict)
            print(predict[0])


            if key == 27: # key 반환 값이 esc(27) 면.. key == ord("q")
                cv2.destroyAllWindows()
                exit(0)


    # cv2.imshow("Lena", img) # 윈도우창 타이틀, 구조체
    # cv2.waitKey(0) # 0은 무한대로 기다리는 것, 입력이 올때까지 이미지를 끄지 않고 기다림, 입력을 하나 주면 창이 닫힘
    # cv2.destroyAllWindows() # 사용한 모든 창을 다 닫아줘. 메모리를 다 돌려준다는 것

