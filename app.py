from flask import Flask, request, jsonify,render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "model.h5"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the model
model = load_model(MODEL_PATH)

# Preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    class_labels = [
        "Speed Limit 20 km/h", "Speed Limit 30 km/h", "Speed Limit 50 km/h", "Speed Limit 60 km/h", "Speed Limit 70 km/h",
        "Speed Limit 80 km/h", "End of Speed Limit 80 km/h", "Speed Limit 100 km/h", "Speed Limit 120 km/h", "No passing",
        "No passing for vehicles over 3.5 metric tons", "Right-of-way at the next intersection", "Priority road", "Yield",
        "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution",
        "Dangerous curve to the left", "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road",
        "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
        "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead",
        "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left", "Keep right", "Keep left",
        "Roundabout mandatory", "End of no passing", "End of no passing by vehicles over 3.5 metric tons"
    ]
    return class_labels[classNo]

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(32, 32), color_mode="grayscale")
    img = np.asarray(img) / 255.0
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    return getClassName(classIndex)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(file_path)

    prediction = model_predict(file_path, model)
    os.remove(file_path)  # Clean up the uploaded file
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
