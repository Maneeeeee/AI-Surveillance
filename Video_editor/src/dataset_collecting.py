import cv2
import os
import time

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Width
cam.set(4, 480)  # Height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_detector.empty():
    print("[ERROR] Haarcascade file not found or failed to load.")
    exit()

face_id = input('\nPlease enter USER ID and press <ENTER>: ')

print("\n[INFO] Starting camera, look at the camera and wait...")

# Create dataset folder if it doesn't exist
user_folder = f'dataset/User.{face_id}'
if not os.path.exists(user_folder):
    os.makedirs(user_folder)
count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to capture image.")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        
        # Save the captured image into the dataset folder
        cv2.imwrite(f"{user_folder}/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)
        time.sleep(0.1)
    k = cv2.waitKey(100) & 0xff
    if k == 27:  # Press 'ESC' to exit
        break
    elif count >= 30:  # Take 30 face samples and stop video
        break

print("[INFO] Dataset collection complete.")
cam.release()
cv2.destroyAllWindows()
