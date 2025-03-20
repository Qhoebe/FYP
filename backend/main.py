import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import pretrained_faster_rcnn_model
from model import ViT_model
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load model once when Flask starts
# model_1 = pretrained_model.initiate_model()
# model_2 = ViT_model.initiate_model()
model = None

@app.route('/evaluate', methods=['POST'])
def count():
    try:
        data = request.json  # Expecting JSON payload
        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        
        global model 
        # Decode Base64 string to OpenCV image
        image_data = data["image"].split(",")[1]  # Remove "data:image/jpeg;base64,"
        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # process image
        count = model.count_objects(image)
        return jsonify({'count': count})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/change_model',methods=['POST'])
def change_model(): 
    try: 
        data = request.json
        if "model" not in data: 
            return jsonify({"error": "No model selected"}), 400
        
        global model 
        model_name = data["model"]
        if model_name == "pretrained_faster_rcnn":
            model = pretrained_faster_rcnn_model.Model()
        elif model_name == "ViT_model":
            model = ViT_model.Model()
        else: 
            return jsonify({'model': "no model selected"})
        
        return jsonify({'model': model_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

if __name__ == "__main__":
    app.run(debug=True)


