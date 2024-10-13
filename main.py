import threading
import cv2
from deepface import Deepface


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIDHT, 480)

counter = 0 
face_match = False
refrence_img = cv2.imread("reference.jpg")

def check_face(fram):
    pass
while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass

        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))


    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()