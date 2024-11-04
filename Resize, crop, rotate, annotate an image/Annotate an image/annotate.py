import cv2

img = cv2.imread("test.png")
cv2.imshow("Original image", img)
cv2.waitKey()

if img is None:
    raise ImportError("Couldn't read image")

# Draw a line
imgLine = img.copy()
pointA = (200, 80)
pointB = (450, 80)
cv2.line(imgLine, pointA, pointB, (255, 255, 255), thickness=40, lineType=cv2.LINE_AA)
cv2.imshow('Image Line', imgLine)
cv2.waitKey()

# Draw a circle
imgCircle = img.copy()
circle_center = (415, 190)
radius = 50
cv2.circle(imgCircle, circle_center, radius, (255, 0, 0), lineType=cv2.LINE_AA)
cv2.imshow("image circle", imgCircle)
cv2.waitKey()

# Draw a filled circle

imgFilled = img.copy()
circle_center = (415, 190)
radius = 55
cv2.circle(imgFilled, circle_center, radius, (255, 30, 0),thickness= -1, lineType=cv2.LINE_8)
cv2.imshow("image with filled circle", imgFilled)
cv2.waitKey()

# Draw a rectangle
imgRectangle = img.copy()
sp = (200, 115)
ep = (450, 225)
cv2.rectangle(imgRectangle, sp, ep, (233, 33, 3), thickness= 5, lineType=cv2.LINE_8)
cv2.imshow("image with rectagle", imgRectangle)
cv2.waitKey()

# Draw an ellipse
imgEllipse = img.copy()
axis1 = (100, 50)
axis2 = (125, 50)

cv2.ellipse(imgEllipse, circle_center, axis1, 0, 0, 360, (255, 255, 80), thickness=5)
cv2.ellipse(imgEllipse, circle_center, axis2, 60, 0, 360, (255, 0, 80), thickness=5)

cv2.imshow('ellipse image',imgEllipse)
cv2.waitKey(0)

cv2.ellipse(imgEllipse, (120, 200), axis1, 0, 30, 180, (255, 255, 80), thickness=5)
cv2.ellipse(imgEllipse, (110, 200), axis2, 60, 0, 180, (255, 0, 80), thickness=-2)

cv2.imshow('half of ellipse image',imgEllipse)
cv2.waitKey(0)

# Putting a text
text = "text for testing"
org = (50, 300)
for i in range(17):
    if i in range(0,8) or i == 16:
        imgText = img.copy()
        cv2.putText(imgText, text, org, fontFace= i, fontScale=i/2, color = (255,2,23))
        cv2.imshow(f"{i} font", imgText)
        cv2.waitKey()
cv2.destroyAllWindows( )
 