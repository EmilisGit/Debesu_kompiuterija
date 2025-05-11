from flask import Flask, request, jsonify
from model import run_inference
import numpy as np
import cv2

ONNX_MODEL_PATH = 'yolo11s.onnx'
CLASS_NAME_PATH = 'class_names.json'

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    try:
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        results = run_inference(image, ONNX_MODEL_PATH, CLASS_NAME_PATH)

        return jsonify({'objects': list(results)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)