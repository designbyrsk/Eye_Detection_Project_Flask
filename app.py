from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)

# Load model
model = load_model("eye_model.keras")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

EYE_CLOSED_THRESHOLD = 3
eye_closed_start = None


def generate_frames():
    global eye_closed_start

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status = "OPEN"

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            resized = cv2.resize(face, (24, 24))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 24, 24, 3))

            prediction = model.predict(reshaped, verbose=0)

            if prediction < 0.5:
                status = "CLOSED"
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, status, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

        if status == "CLOSED":
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > EYE_CLOSED_THRESHOLD:
                status = "ALERT"
        else:
            eye_closed_start = None

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(debug=True)