import serial
import torch
import cv2
import time
import face_recognition
import numpy as np
import os


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


# loading thw known faces
known_face_encodings = []
known_face_names = []

def load_known_faces(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            image = face_recognition.load_image_file(filepath)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])

load_known_faces('known_faces')


process_frame_interval = 2
frame_count = 0


while True:
    ret, frame = camera.read()
    if not ret:
        print("[-]Failed to grab frame!")
        break

    frame_count += 1
    if frame_count % process_frame_interval != 0:
        continue

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
        
        if model.names[int(cls)] == "person":
            # extract the ROI of the person
            top, left, bottom, right = xyxy[1], xyxy[0], xyxy[3], xyxy[2]
            roi = frame[top:bottom, left:right]
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) # converting to RGB
            face_locations = face_recognition.face_locations(rgb_roi)
            face_encodings = face_recognition.face_encodings(rgb_roi, face_locations)
            for face_encoding in face_encodings:
                # See if the face is a match for the known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                print(name)



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
