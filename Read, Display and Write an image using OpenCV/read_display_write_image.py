import cv2

img_color = cv2.imread('test.png', 1)
img_grayscale = cv2.imread('test.png', 0)
img_unchanged = cv2.imread('test.png', -1)

cv2.imshow("color",img_color)
cv2.imshow("grayscale",img_grayscale)
cv2.imshow("unchanged",img_unchanged)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

cv2.imwrite('test.png',img_color)
cv2.destroyAllWindows


