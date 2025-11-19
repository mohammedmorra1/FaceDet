from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import requests
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow requests from your React app

mp_face_detection = mp.solutions.face_detection

def url_to_image(url):
    resp = requests.get(url, stream=True).content
    img = np.asarray(bytearray(resp), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

@app.route("/detect_faces", methods=["POST"])
def detect_faces():
    data = request.get_json()
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    image = url_to_image(image_url)
    h, w, _ = image.shape

    results_data = []
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                results_data.append({
                    "left_col": bbox.xmin,
                    "top_row": bbox.ymin,
                    "right_col": bbox.xmin + bbox.width,
                    "bottom_row": bbox.ymin + bbox.height
                })

    return jsonify(results_data)
@app.route("/", methods=["GET"])
def index():
    return "Face Detection API is running."

port = int(os.environ.get("PORT", 5000))
print(f"Server running on port {port}")
app.run(host="0.0.0.0", port=port)