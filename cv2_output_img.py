import cv2

# 이미지 불러오기 image는 numpy.ndarray 타입
image = cv2.imread("images/lunar.jpg", cv2.IMREAD_ANYCOLOR)

# 창을 띄워서 이미지 출력
cv2.imshow("Moon", image)
# 키 입력을 기다림
cv2.waitKey()
cv2.destroyAllWindows()