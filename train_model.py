import cv2
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound
import time

# ----------------- Load Model -----------------
model = load_model("eye_model.keras")

# ----------------- Load Haar cascade for eyes -----------------
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ----------------- Setup -----------------
cap = cv2.VideoCapture(0)
EYE_CLOSED_THRESHOLD = 3  # seconds
eye_closed_start = None
ALERT_SOUND = "alert.mp3"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    closed_eyes = 0

    for (ex, ey, ew, eh) in eyes:
        eye_img = gray[ey:ey+eh, ex:ex+ew]
        eye_img = cv2.resize(eye_img, (24,24))
        eye_img = eye_img / 255.0
        eye_img = np.reshape(eye_img, (1,24,24,1))

        prediction = model.predict(eye_img, verbose=0)
        if prediction < 0.5:  # Eye is closed
            closed_eyes += 1

        # Draw rectangle around eyes
        color = (0,0,255) if prediction<0.5 else (0,255,0)
        cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), color, 2)

    # If no eyes detected, assume closed
    if len(eyes)==0:
        closed_eyes += 2  # To trigger alert if both eyes not detected

    # Check for continuous eye closure
    if closed_eyes >= 2:
        if eye_closed_start is None:
            eye_closed_start = time.time()
        else:
            elapsed = time.time() - eye_closed_start
            if elapsed > EYE_CLOSED_THRESHOLD:
                playsound(ALERT_SOUND)
                cv2.putText(frame, "WAKE UP!", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                eye_closed_start = None
    else:
        eye_closed_start = None

    cv2.imshow("Eye Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()