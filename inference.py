import serial
import torch
import cv2
import time

# Load the trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

# Initialize the webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open video stream from webcam")
    exit()

# Initialize serial communication with Arduino
ser = serial.Serial('COM3', 9600, timeout=1)  # Adjust serial port as necessary

# Allow some time for the camera to warm up
time.sleep(2)

def read_distance():
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            return int(line)
    except ValueError:
        print("[-]Some error occured in calculating distance.")
        return None
    return None

while True:
    ret, frame = camera.read()
    if not ret:
        print("[-]Failed to grab frame!")
        break

    # Perform object detection
    results = model(frame)

    # Extract bounding boxes and labels
    for *xyxy, conf, cls in results.xyxy[0]:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        xyxy = [int(x) for x in xyxy]

        # Read distance data from Arduino
        distance_mm = read_distance()
        if distance_mm is not None:
            distance_in = distance_mm / 25.4  # Convert to inches
            label += f' | Distance: {distance_in:.2f} inches'
            print(f"A {model.names[int(cls)]} is at a distance {distance_in:.2f} inches from you")

        # Draw bounding box and label on the frame
        frame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        frame = cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam Image', frame)

    # Add a break condition (Press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
camera.release()
cv2.destroyAllWindows()
