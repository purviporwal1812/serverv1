from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
import os
import uuid
import json
import numpy as np
import shutil
import kagglehub
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin

# Read critical configuration from environment variables.
KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET")
if not KAGGLE_DATASET:
    raise ValueError("KAGGLE_DATASET environment variable not set!")

# Non-critical configuration with defaults.
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
ALLOWED_EXTENSIONS = set(os.environ.get("ALLOWED_EXTENSIONS", "png,jpg,jpeg,gif").split(","))

# IMAGE_SIZE should be provided as "width,height" (e.g., "224,224")
IMAGE_SIZE_STR = os.environ.get("IMAGE_SIZE", "224,224")
try:
    IMAGE_SIZE = tuple(int(x.strip()) for x in IMAGE_SIZE_STR.split(","))
    if len(IMAGE_SIZE) != 2:
        raise ValueError()
except Exception:
    raise ValueError("IMAGE_SIZE environment variable must be in the format 'width,height'")

PORT = int(os.environ.get("PORT", "10000"))

# Initialize Flask app and enable CORS.
app = Flask(__name__)
CORS(app)

# Configure upload folder.
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create models directory if it doesn't exist.
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, 'final_model.keras')
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, 'class_indices.json')

def download_dataset():
    """
    Uses kagglehub to download the dataset specified by KAGGLE_DATASET.
    Returns the path where the dataset files are extracted.
    """
    try:
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print("Path to dataset files:", dataset_path)
        return dataset_path
    except Exception as e:
        print("Error downloading dataset:", e)
        return None

def download_model():
    """
    Downloads the model file from the Kaggle dataset and saves it to MODEL_PATH.
    """
    if not os.path.exists(MODEL_PATH):
        dataset_path = download_dataset()
        if not dataset_path:
            with open(f"{MODEL_PATH}.failed", "w") as f:
                f.write("Dataset download failed.")
            return False
        src_model = os.path.join(dataset_path, "final_model.keras")
        if not os.path.exists(src_model):
            error_msg = f"Model file 'final_model.keras' not found in dataset path: {dataset_path}"
            print(error_msg)
            with open(f"{MODEL_PATH}.failed", "w") as f:
                f.write(error_msg)
            return False
        try:
            shutil.copy(src_model, MODEL_PATH)
            print(f"Model copied to {MODEL_PATH}")
            return True
        except Exception as e:
            print("Error copying model file:", e)
            with open(f"{MODEL_PATH}.failed", "w") as f:
                f.write(f"Copy failed: {str(e)}")
            return False
    else:
        print("Model already exists locally.")
        return True

def download_class_indices():
    """
    Downloads the class indices file from the Kaggle dataset and saves it to CLASS_INDICES_PATH.
    """
    if not os.path.exists(CLASS_INDICES_PATH):
        dataset_path = download_dataset()
        if not dataset_path:
            with open(f"{CLASS_INDICES_PATH}.failed", "w") as f:
                f.write("Dataset download failed.")
            return False
        src_class_indices = os.path.join(dataset_path, "class_indices.json")
        if not os.path.exists(src_class_indices):
            error_msg = f"Class indices file 'class_indices.json' not found in dataset path: {dataset_path}"
            print(error_msg)
            with open(f"{CLASS_INDICES_PATH}.failed", "w") as f:
                f.write(error_msg)
            return False
        try:
            shutil.copy(src_class_indices, CLASS_INDICES_PATH)
            print(f"Class indices copied to {CLASS_INDICES_PATH}")
            return True
        except Exception as e:
            print("Error copying class indices file:", e)
            with open(f"{CLASS_INDICES_PATH}.failed", "w") as f:
                f.write(f"Copy failed: {str(e)}")
            return False
    else:
        print("Class indices already exists locally.")
        return True

# Attempt to download files at startup.
model_download_success = download_model()
class_indices_download_success = download_class_indices()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_species(image_path, model_path=None, class_indices_path=None):
    """
    Function for making predictions.
    
    Parameters:
        image_path (str): Path to the image file.
        model_path (str): Path to the saved model (optional).
        class_indices_path (str): Path to saved class indices JSON (optional).
    
    Returns:
        dict: Prediction results including species and confidence.
    """
    if model_path is None:
        model_path = MODEL_PATH
    if class_indices_path is None:
        class_indices_path = CLASS_INDICES_PATH

    # Check if model and class indices are available
    if not os.path.exists(model_path):
        return {"error": "Model file not available. Please check server logs."}
    if not os.path.exists(class_indices_path):
        return {"error": "Class indices file not available. Please check server logs."}

    # Load the model
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"error": f"Could not load model: {str(e)}"}

    # Load class indices
    try:
        with open(class_indices_path, 'r') as f:
            class_indices_str = json.load(f)
            class_indices = {k: v for k, v in class_indices_str.items()}
    except Exception as e:
        print(f"Error loading class indices: {e}")
        return {"error": f"Could not load class indices: {str(e)}"}

    # Load and preprocess image
    try:
        img = image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": f"Could not process image: {str(e)}"}

    # Predict
    try:
        prediction = model.predict(img_array)
        predicted_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_idx])

        # Convert index to class name
        class_names = {int(v): k for k, v in class_indices.items()}
        species = class_names[predicted_idx]

        result = {
            'species': species,
            'confidence': confidence,
            'top_predictions': [
                {
                    'species': class_names[int(i)],
                    'confidence': float(prediction[0][i])
                }
                for i in np.argsort(-prediction[0])[:5]  # Top 5 predictions
            ]
        }
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}

@app.route('/')
def home():
    # Add status check for model and class indices
    model_status = "Available" if os.path.exists(MODEL_PATH) else "Not Available"
    class_indices_status = "Available" if os.path.exists(CLASS_INDICES_PATH) else "Not Available"
    
    return render_template('index.html', 
                           model_status=model_status,
                           class_indices_status=class_indices_status)

@app.route('/status')
def status():
    """Endpoint to check the status of model and class indices files"""
    model_available = os.path.exists(MODEL_PATH)
    class_indices_available = os.path.exists(CLASS_INDICES_PATH)
    
    # Check for failed downloads
    model_failed = os.path.exists(f"{MODEL_PATH}.failed")
    class_indices_failed = os.path.exists(f"{CLASS_INDICES_PATH}.failed")
    
    # Read error messages if available
    model_error = None
    class_indices_error = None
    
    if model_failed:
        with open(f"{MODEL_PATH}.failed", "r") as f:
            model_error = f.read()
    if class_indices_failed:
        with open(f"{CLASS_INDICES_PATH}.failed", "r") as f:
            class_indices_error = f.read()
    
    return jsonify({
        "model_available": model_available,
        "class_indices_available": class_indices_available,
        "model_failed": model_failed,
        "class_indices_failed": class_indices_failed,
        "model_error": model_error,
        "class_indices_error": class_indices_error
    })

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    # Check if model and class indices are available
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not available. Please check server status at /status endpoint."})
    if not os.path.exists(CLASS_INDICES_PATH):
        return jsonify({"error": "Class indices not available. Please check server status at /status endpoint."})
    
    # Allow GET for testing /predict separately
    if request.method == 'GET':
        return jsonify({"message": "Predict endpoint is accessible."})
    
    # POST method for file upload and prediction
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        prediction_result = predict_species(
            image_path=file_path,
            model_path=MODEL_PATH,
            class_indices_path=CLASS_INDICES_PATH
        )
        
        return jsonify(prediction_result)
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/trigger-download', methods=['POST'])
def trigger_download():
    """Endpoint to manually trigger model and class indices download"""
    try:
        model_result = download_model()
        class_indices_result = download_class_indices()
        
        return jsonify({
            "model_download_success": model_result,
            "class_indices_download_success": class_indices_result
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
