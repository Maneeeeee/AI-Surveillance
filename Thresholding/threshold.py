import cv2

img = cv2.imread("image.jpg")


th, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
cv2.imwrite("threshold_example.jpg", dst)
cv2.imshow("",dst)
cv2.waitKey()

th, dst = cv2.threshold(img, 0, 128, cv2.THRESH_BINARY)
cv2.imwrite("threshold_bin_maxval.jpg", dst)
cv2.imshow("",dst)
cv2.waitKey()

th, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("threshold_bin.jpg", dst)
cv2.imshow("",dst)
cv2.waitKey()

th, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("threshold_bin_inv.jpg", dst)
cv2.imshow("",dst)
cv2.waitKey()

th, dst = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
cv2.imwrite("threshold_trunc.jpg", dst)
cv2.imshow("",dst)
cv2.waitKey()

th, dst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
cv2.imwrite("threshold_tozero.jpg", dst)
cv2.imshow("",dst)
cv2.waitKey()

th, dst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imwrite("threshold_ttozero_inv.jpg", dst)
cv2.imshow("",dst)
cv2.waitKey()
cv2.destroyAllWindows()