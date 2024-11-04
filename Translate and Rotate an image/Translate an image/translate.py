import cv2
import numpy as np

image = cv2.imread("download.png")
h,w = image.shape[:2]

tx, ty = w / 4, h/ 4
translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
translated_image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(w,h))
cv2.imshow('Translated image', translated_image)
cv2.imshow('Original image', image)
cv2.waitKey(0)
cv2.imwrite('translated_image.jpg', translated_image)