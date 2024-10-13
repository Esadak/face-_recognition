import threading
import cv2
import os
from deepface import DeepFace

# Starta kameran
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Globala variabler
counter = 0 
face_match = False
matched_name = ""
reference_images_folder = "image"  # Mappen med referensbilder

# Läs referensbilder och namnge dem
def load_reference_images(folder):
    reference_images = {}
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Kontrollera filtyper
            name = filename.split('.')[0]  # Namnet baserat på filnamnet
            image_path = os.path.join(folder, filename)
            reference_images[name] = cv2.imread(image_path)  # Lägg till bilden
    return reference_images

# Kolla ansikten
def check_face(frame, reference_images):
    global face_match, matched_name
    try:
        for name, reference_img in reference_images.items():
            # Verifiera med varje referensbild
            if DeepFace.verify(frame, reference_img.copy())['verified']:
                face_match = True
                matched_name = name
                return  # Sluta när vi hittar en match
        face_match = False
        matched_name = ""
    except ValueError:
        face_match = False
        matched_name = ""

# Ladda referensbilder
reference_images = load_reference_images(reference_images_folder)

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:  # Kontrollera ansikten var 30:e frame
            threading.Thread(target=check_face, args=(frame.copy(), reference_images)).start()

        counter += 1

        # Visa resultatet
        if face_match:
            cv2.putText(frame, f"MATCH: {matched_name}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
