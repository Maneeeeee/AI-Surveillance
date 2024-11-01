import cv2
import os

print(os.path.exists('test.mp4'))
 
vid_captured = cv2.VideoCapture("test.mp4")

if vid_captured.isOpened() == False:
    print("Video file can't open")
else:
    fsp = int(vid_captured.get(5))
    print("Frame Rate",fsp,"frames per second")

    frame_count = vid_captured.get(7)
    print("Frame count : ", frame_count)

while vid_captured.isOpen():
  ret, frame = vid_captured.read()
  if ret == True:
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(200)
    if k == 113:
      break
    else:
      break

vid_captured.release()
cv2.destroyAllWindows()
