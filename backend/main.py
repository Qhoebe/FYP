import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import pretrained_model, ViT_model
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load model once when Flask starts
model_1 = pretrained_model.initiate_model()
# model_2 = ViT_model.initiate_model()

@app.route('/evaluate/model_1', methods=['POST'])
def count_model1():
    try:
        data = request.json  # Expecting JSON payload
        if "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode Base64 string to OpenCV image
        image_data = data["image"].split(",")[1]  # Remove "data:image/jpeg;base64,"
        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # process image
        count = pretrained_model.count_objects(model_1, image)
        return jsonify({'count': count})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# @app.route('/evaluate/model_2', methods=['POST'])
# def count_model2():
#     try:
#         data = request.json  # Expecting JSON payload
#         if "image" not in data:
#             return jsonify({"error": "No image provided"}), 400
        
#         # Decode Base64 string to OpenCV image
#         image_data = data["image"].split(",")[1]  # Remove "data:image/jpeg;base64,"
#         decoded_image = base64.b64decode(image_data)
#         np_arr = np.frombuffer(decoded_image, np.uint8)
#         image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = Image.fromarray(image)

#         # process image
#         count = ViT_model.count_objects(model_2, image)
#         return jsonify({'count': count})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)


