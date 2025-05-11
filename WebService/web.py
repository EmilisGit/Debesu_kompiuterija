from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

YOLO_MODEL_URL = os.getenv("YOLO_MODEL_URL", "http://127.0.0.1:5000")

@app.route("/", methods=["GET"])
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/", methods=["POST"])
def upload():
    try:
        result = "failed"
        file = request.files["image"]
        response = requests.post(f'{YOLO_MODEL_URL}/predict', files={"image":file})
        result = response.json()
        return jsonify({'objects': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)