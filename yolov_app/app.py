from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure model file exists in the directory

# Open webcam
cap = cv2.VideoCapture(0)

# Global variable for person count
person_count = 0

def generate_frames():
    global person_count
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Run YOLOv8 inference
        results = model(frame)

        # Reset person count
        person_count = 0

        # Draw detections
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                
                # Filter only "person" category (class index = 0)
                if cls == 0:
                    person_count += 1  # Increase count for each detected person
                    label = f"Person: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/person_count')
def get_person_count():
    """Returns the number of detected people."""
    return jsonify({'count': person_count})

if __name__ == "__main__":
    app.run(debug=True)
