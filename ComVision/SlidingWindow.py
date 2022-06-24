import cv2

if __name__ == '__main__':
    img = cv2.imread("Lenna.png") # 이미지 읽어오는 거
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    w_width = 200
    w_height = 200

    # y도, x도 100까지 움직이게 할거야
    for y in range(100):
        for x in range(100):
            crop = gray[y:y+w_width, x:x+w_width]
            cv2.imshow("crop_img",crop)
            key = cv2.waitKey(33)

            if key == 27: # key 반환 값이 esc(27) 면.. key == ord("q")
                cv2.destroyAllWindows()
                exit(0)


    cv2.imshow("Lena", img) # 윈도우창 타이틀, 구조체
    cv2.waitKey(0) # 0은 무한대로 기다리는 것, 입력이 올때까지 이미지를 끄지 않고 기다림, 입력을 하나 주면 창이 닫힘
    cv2.destroyAllWindows() # 사용한 모든 창을 다 닫아줘. 메모리를 다 돌려준다는 것

