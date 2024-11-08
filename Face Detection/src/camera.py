import cv2
class Camera:
    def __init__(self, camera_index = 0):
        self.cap = cv2.VideoCapture(camera_index)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed reading frame")
        return frame
    
    def releade(self):
        self.cap.release()
        