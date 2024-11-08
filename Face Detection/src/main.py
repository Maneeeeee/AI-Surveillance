import cv2
import os
from camera import Camera
from face_detection import FaceDetector

cascade_path = "data/haarcascade_frontalface_default.xml"

if not os.path.exists(cascade_path):
    print("Error: Cascade file's not found ")
    exit(1)

face_detector = FaceDetector(cascade_path)
act_width = 18.0
focal_length = 500.0
camera = Camera()

try :
    while True:
        frame = camera.read_frame()
        faces = face_detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x + w, y + h), (68, 68, 68), 2)
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF in {ord('q'), 27}:
            break


except RuntimeError as e:
    exit(1)
finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Program terminated gracefully.")
