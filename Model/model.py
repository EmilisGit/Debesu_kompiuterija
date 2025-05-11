import onnxruntime as ort
import numpy as np
import cv2
import json

def run_inference(image: np.ndarray, onnx_model_path: str, class_name_path: str):

    with open(class_name_path, 'r') as file:
        class_names = json.load(file)
    class_names = {int(k): v for k, v in class_names.items()}

    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).reshape(1, 3, 640, 640)
    img = img / 255.0
    img = img.astype(np.float32)

    outputs = session.run(None, {input_name: img})
    results = outputs[0]
    results = results.transpose()

    A = []
    for detection in results:
        class_id = detection[4:].argmax()
        confidence_score = detection[4:].max()
        new_detection = np.append(detection[:4], [class_id, confidence_score])
        A.append(new_detection)
    A = np.array(A)

    predictions = np.array([d for d in A if d[-1] > 0.2]) 

    names = []
    for pred in predictions:
        class_id = int(pred[-2])
        names.append((class_names.get(class_id, "Unknown"), pred[-1]))

    sorted_results = sorted(names, key=lambda x: x[1], reverse=True)
    top_results = set([x[0] for x in sorted_results])

    return top_results
