from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load AI model
model = load_model("eye_model.keras", compile=False)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    # decode image
    img_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # preprocess
    resized = cv2.resize(frame, (24, 24))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 24, 24, 3))

    prediction = model.predict(reshaped, verbose=0)

    if prediction < 0.5:
        status = "CLOSED"
    else:
        status = "OPEN"

    return jsonify({"status": status})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)