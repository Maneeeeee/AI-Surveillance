import cv2
import os
import time
from face_detection import FaceDetector
from training import recognizer

cascade_path = "data/haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print("Error: Cascade file's not found ")
    exit(1)

face_detector = FaceDetector(cascade_path)
act_width = 18.0
focal_length = 500.0

current_effect = None
cap = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 60.0  # Confidence threshold for recognition
MAX_CONFIDENCE = 200.0       # Maximum confidence considered for 0% similarity

current_effect = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("can not cap")
            break
        # frame = cap.read_frame()  # Capture a frame from the camera
        faces = face_detector.detect_faces(frame)  # Detect faces in the frame

        for (x, y, w, h) in faces:
            face_region = frame[y:y + h, x:x + w]
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (200, 200))  # Resize for consistency

            # Recognize face using LBPH model
            label, confidence = recognizer.predict(resized_face)

            if confidence < CONFIDENCE_THRESHOLD:
                # Recognized person: calculate similarity and display
                similarity = max(0, 100 - (confidence / MAX_CONFIDENCE * 100))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {label} Similarity: {similarity:.2f}%", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Not recognized: draw bounding box and display "Not recognized"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for unrecognized faces
                cv2.putText(frame, "Not recognized", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, str(time.ctime()), (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if current_effect == 'v':
                frame = cv2.flip(frame, 1)
            elif current_effect == 'h':
                frame = cv2.flip(frame, 0)
            elif current_effect == 's':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            elif current_effect == 'l':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            elif current_effect == 'b':
                frame = cv2.GaussianBlur(frame, (7, 7), 0)
            elif current_effect == 'n':
                frame = cv2.medianBlur(frame, 7)
            elif current_effect == 'd':
                frame = cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 50, (0, 255, 0), 2)
            elif current_effect == 'r':
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif current_effect == 'g':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for display consistency

            # Display the frame
            cv2.imshow("Real - time video", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('v'):
                current_effect = 'v'
            elif key == ord('h'):
                current_effect = 'h'
            elif key == ord('s'):
                current_effect = 's'
                frame_color = "hsv"
            elif key == ord('l'):
                current_effect = 'l'
                frame_color = "lab"
            elif key == ord('b'):
                current_effect = 'b'
            elif key == ord('n'):
                current_effect = 'n'
            elif key == ord('d'):
                current_effect = 'd'
            elif key == ord('r'):
                current_effect = 'r'
            elif key == ord('g'):
                current_effect = 'g'
                frame_color = "g"
            elif key == ord('c'): 
                current_effect = None
            elif key in {ord('q'), 27}:  
                cap.release()
                cv2.destroyAllWindows()

except RuntimeError as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()

print("Program terminated gracefully.")



