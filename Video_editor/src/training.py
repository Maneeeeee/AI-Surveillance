# training

import cv2
import os
import numpy as np
from face_detection import FaceDetector

cascade_path = "data/haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print("Error: Cascade file's not found ")
    exit(1)

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

recognizer.save("src/face_recognizer.yml")
print("Training comlete and model saved. ")

