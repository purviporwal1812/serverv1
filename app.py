from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
import os
import uuid
import json
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from flask_cors import CORS, cross_origin

# Load TFLite model once
TFLITE_MODEL_PATH = os.path.join("models", "final_model.tflite")
CLASS_INDICES_PATH = os.path.join("models", "class_indices.json")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read critical configuration
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = set(os.environ.get("ALLOWED_EXTENSIONS", "png,jpg,jpeg,gif").split(","))
IMAGE_SIZE_STR = os.environ.get("IMAGE_SIZE", "224,224")
PORT = int(os.environ.get("PORT", "10000"))

try:
    IMAGE_SIZE = tuple(int(x.strip()) for x in IMAGE_SIZE_STR.split(","))
    if len(IMAGE_SIZE) != 2:
        raise ValueError()
except Exception:
    raise ValueError("IMAGE_SIZE environment variable must be in the format 'width,height'")

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_species(image_path, class_indices_path=None):
    if class_indices_path is None:
        class_indices_path = CLASS_INDICES_PATH

    if not os.path.exists(class_indices_path):
        return {"error": "Class indices file not available. Please check server logs."}

    # Load class indices
    try:
        with open(class_indices_path, 'r') as f:
            class_indices_str = json.load(f)
            class_indices = {int(v): k for k, v in class_indices_str.items()}
    except Exception as e:
        return {"error": f"Could not load class indices: {str(e)}"}

    # Preprocess image
    try:
        img = image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return {"error": f"Could not process image: {str(e)}"}

    # TFLite prediction
    try:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0]

        predicted_idx = int(np.argmax(prediction))
        confidence = float(prediction[predicted_idx])

        result = {
            'species': class_indices[predicted_idx],
            'confidence': confidence,
            'top_predictions': [
                {
                    'species': class_indices[int(i)],
                    'confidence': float(prediction[i])
                } for i in np.argsort(-prediction)[:5]
            ]
        }
        return result
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.route('/')
def home():
    model_status = "Available" if os.path.exists(TFLITE_MODEL_PATH) else "Not Available"
    class_indices_status = "Available" if os.path.exists(CLASS_INDICES_PATH) else "Not Available"
    return render_template('index.html', model_status=model_status, class_indices_status=class_indices_status)


@app.route('/status')
def status():
    model_available = os.path.exists(TFLITE_MODEL_PATH)
    class_indices_available = os.path.exists(CLASS_INDICES_PATH)
    return jsonify({
        "model_available": model_available,
        "class_indices_available": class_indices_available,
        "model_failed": False,
        "class_indices_failed": False,
        "model_error": None,
        "class_indices_error": None
    })


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    try:
        if request.method == 'GET':
            return jsonify({"message": "Predict endpoint is accessible."})

        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            prediction_result = predict_species(image_path=file_path)

            if 'error' in prediction_result:
                return jsonify({'error': prediction_result['error']}), 500

            return jsonify(prediction_result), 200

        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
