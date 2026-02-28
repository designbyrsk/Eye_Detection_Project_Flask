import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("eye_model.keras")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (24, 24))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 24, 24, 3))

        prediction = model.predict(reshaped, verbose=0)

        if prediction > 0.5:
            label = "OPEN"
            color = (0, 255, 0)
        else:
            label = "CLOSED"
            color = (0, 0, 255)

        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Eye Detection", frame)

    if cv2.waitKey(1) == 27:   # ESC key
        break

cap.release()
cv2.destroyAllWindows()