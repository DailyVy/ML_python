import cv2
import glob

imgsrc = glob.glob("D:/ML_python/ComVision/*.png")
# print(imgsrc)
src = imgsrc[0]

if __name__ == "__main__":
    img = cv2.imread(src)
    # cv2.imshow("Lena", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    w_width = 200
    w_height = 200

    for y in range(100):
        for x in range(100):
            crop = gray[y:y+w_height, x:x+w_width]
            print(f"({x}, {y}) to ({x+w_width}, {y+w_height})")
            cv2.imshow("crop_img", crop)
            key = cv2.waitKey(33) # 0을 넣으면 key입력까지 무한대 대기 ==> 입력하면 움직임
            # 33 => 33ms 동안 대기하다가 key 입력이 없으면 그 다음거 수행

            # 근데 33ms 안에 key를 넣어줬다면? waitKey()에 key가 리턴됨 ==> callback
            # 반환된 key값은 ASCII코드, 27은 esc
            # if key == 27:
            if key == ord("q"): # q 누르면 꺼짐
                cv2.destroyAllWindows()
                exit(0)

    cv2.imshow("Lena", img)  # 윈도우창 타이틀, 구조체
    cv2.waitKey(0)
    cv2.destroyAllWindows()

