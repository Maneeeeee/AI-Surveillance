import cv2
import os
import numpy as np
from face_detection import FaceDetector
from camera import Camera

cascade_path = "haar_data/haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print("Error: Cascade file's not found ")
    exit(1)

camera = Camera()

face_detector = FaceDetector(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = "dataset"
training_faces = []
labels = []


for person_label, person_name in enumerate(os.listdir(dataset_path)):
    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        
        frame = cv2.imread(image_path)
        faces = face_detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            face_region = frame[y:y + h, x:x + w]
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (200, 200))

            training_faces.append(resized_face)
            labels.append(person_label)
    
recognizer.train(training_faces, np.array(labels))

recognizer.save("face_recognizer.yml")
print("Training comlete and model saved. ")


# Set parameters
CONFIDENCE_THRESHOLD = 60.0  # Confidence threshold for recognition
MAX_CONFIDENCE = 100.0       # Maximum confidence considered for 0% similarity

try:
    while True:
        frame = camera.read_frame()  # Capture a frame from the camera
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

        # Display the frame
        cv2.imshow("Real-time Face Recognition", frame)

        # Exit if 'q' or 'Esc' is pressed
        if cv2.waitKey(1) & 0xFF in {ord('q'), 27}:
            break

except RuntimeError as e:
    print(f"Error: {e}")
finally:
    camera.release()
    cv2.destroyAllWindows()
