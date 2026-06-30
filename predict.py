"""
predict.py
Loads the trained model and predicts the soil nutrient class for a single image.

Usage:
    python predict.py path/to/image.jpg
"""

import sys

import cv2
import numpy as np
import tensorflow as tf

from data_loader import CLASS_NAMES, IMAGE_SIZE

MODEL_PATH = "models/trained_model.h5"


def predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")

    return predicted_class, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    predict(sys.argv[1])
