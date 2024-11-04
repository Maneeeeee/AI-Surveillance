import cv2
import numpy as np

image = cv2.imread("download.png")
h, w = image.shape[:2]
c = (w/2, h/2)

rotated_matrix = cv2.getRotationMatrix2D(center=c, angle= -60, scale= 1)
rotated_image = cv2.warpAffine(src=image, M= rotated_matrix, dsize = (w, h))

cv2.imshow("Original image", image)
cv2.imshow("Rotated image", rotated_image)
cv2.waitKey()
rotated_matrix = cv2.getRotationMatrix2D(center=c, angle= 60, scale= 1)
rotated_image = cv2.warpAffine(src=image, M= rotated_matrix, dsize = (w, h))

cv2.imshow("Original image", image)
cv2.imshow("Rotated image", rotated_image)
cv2.imwrite("rotated_image.jpg", rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()

