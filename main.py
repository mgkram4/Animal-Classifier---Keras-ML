import logging
import os
import sys

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


# Custom DepthwiseConv2D class to handle the unrecognized 'groups' argument
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)


# Register the custom layer with Keras
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({"DepthwiseConv2D": CustomDepthwiseConv2D})


def load_model_safely(model_path):
    try:
        # Try loading with custom objects
        model = load_model(
            model_path, custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
        )
        logging.info("Model loaded successfully with custom objects.")
    except Exception as e:
        logging.warning(f"Failed to load model with custom objects: {e}")
        try:
            # Try loading without custom objects
            model = load_model(model_path)
            logging.info("Model loaded successfully without custom objects.")
        except Exception as e:
            logging.error(f"Failed to load model without custom objects: {e}")
            try:
                # Try loading as HDF5 file
                import h5py

                with h5py.File(model_path, "r") as f:
                    model = load_model(f)
                logging.info("Model loaded successfully as HDF5 file.")
            except Exception as e:
                logging.error(f"Failed to load model as HDF5 file: {e}")
                raise
    return model


# Load the trained model
try:
    model_path = os.path.abspath("keras_model.h5")
    model = load_model_safely(model_path)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    sys.exit(1)

# Load the labels
try:
    labels_path = "labels.txt"
    with open(labels_path, "r") as file:
        labels = file.read().splitlines()
    logging.info("Labels loaded successfully.")
except Exception as e:
    logging.error(f"Error loading labels: {e}")
    sys.exit(1)


def preprocess_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise


def predict_image(image_path):
    try:
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
        predicted_label = labels[np.argmax(predictions)]
        return predicted_label, predictions
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise


# Initialize Flask app
app = Flask(__name__)

# Set up the upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            predicted_label, predictions = predict_image(filepath)

            os.remove(filepath)

            return jsonify(
                {
                    "predicted_label": predicted_label,
                    "raw_predictions": predictions.tolist(),
                }
            )
    except Exception as e:
        logging.error(f"Error in /predict route: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
