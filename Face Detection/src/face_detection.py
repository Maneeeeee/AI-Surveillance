import cv2

class FaceDetector:
    def __init__(self, cascade_path, scale_factor =1.1, min_neighbors = 5, min_size=(30, 30)):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError
        
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor = self.scale_factor, minNeighbors = self.min_neighbors, minSize = self.min_size)
        return faces