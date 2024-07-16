from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np

app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

def detect_objects(frame):
    # Perform object detection
    results = model(frame)

    # Render results on the frame
    frame = results.render()[0]

    # Convert frame to JPEG image
    _, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

def gen(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        processed_frame = detect_objects(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Initialize the webcam
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return "Error: Could not open video stream from webcam"

    # Stream the video feed with object detection
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
