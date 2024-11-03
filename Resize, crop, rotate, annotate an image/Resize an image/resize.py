import cv2
import numpy as np

image = cv2.imread("test.png")
h, w, c = image.shape
print(h,w,c)


# resize 2 times smaller
down_height = 96
down_width = 102
down_points = (down_width, down_height)
resize_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR) 
cv2.imshow("resized smaller", resize_down)
cv2.waitKey()

# resized 2 times bigger
down_height = 384
down_width = 408
down_points = (down_width, down_height)
resize_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR) 
cv2.imshow("resized bigger", resize_down)
cv2.waitKey()
cv2.imshow("original", image)
cv2.waitKey()
scale_up_x = 1.2
scale_up_y = 1.2

scale_down = 0.6
scaled_f_down = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
scaled_f_up = cv2.resize(image, None, fx= scale_up_x, fy= scale_up_y, interpolation= cv2.INTER_LINEAR)
cv2.imshow("scaled down", scaled_f_down)
cv2.waitKey()
cv2.imshow("scaled up", scaled_f_up)
cv2.waitKey(0)

scale_down = 0.5
res_inter_nearest = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_NEAREST)
res_inter_linear = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
res_inter_area = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_AREA)
cv2.imshow("inter nearest", res_inter_nearest)
cv2.waitKey()
cv2.imshow("inter liear", res_inter_linear)
cv2.waitKey(0)
cv2.imshow("inter area", res_inter_area)
cv2.waitKey(0)


if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


cv2.destroyAllWindows()