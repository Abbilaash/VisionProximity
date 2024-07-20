import serial
import torch
import cv2
import time
import pyttsx3
import threading
import speech_recognition as sr

# Initialize pyttsx3
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

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
        print("[-] Some error occurred in calculating distance.")
        return None
    return None

def speech_thread():
    while True:
        if message_queue:
            message = message_queue.pop(0)
            speak(message)
        time.sleep(1)  # Small delay to prevent busy waiting

def voice_input_thread():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        with microphone as source:
            print("Listening for user commands...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio)
            print(f"User said: {command}")

            # Add your custom voice commands handling here
            if "what is this" in command.lower():
                if last_object:
                    speak(f"The object in front of you is {last_object}")

        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

# Create a queue for messages
message_queue = []

# Start the speech thread
threading.Thread(target=speech_thread, daemon=True).start()

# Start the voice input thread
threading.Thread(target=voice_input_thread, daemon=True).start()

# Initialize variables for tracking
last_distance = None
last_object = None
last_speak_time = 0
speak_interval = 5  # seconds

while True:
    ret, frame = camera.read()
    if not ret:
        print("[-] Failed to grab frame!")
        break

    # Perform object detection
    results = model(frame)

    # Extract bounding boxes and labels
    closest_object = None
    closest_distance = float('inf')
    frame_center_x = frame.shape[1] / 2

    for *xyxy, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        xyxy = [int(x) for x in xyxy]

        # Calculate center of the bounding box
        bbox_center_x = (xyxy[0] + xyxy[2]) / 2
        center_distance = abs(frame_center_x - bbox_center_x)

        # Read distance data from Arduino
        distance_mm = read_distance()
        if distance_mm is not None:
            distance_in = distance_mm / 25.4  # Convert to inches
            if center_distance < closest_distance:
                closest_distance = center_distance
                closest_object = (label, distance_in)

        # Draw bounding box and label on the frame
        frame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        frame = cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Check and speak the closest object
    if closest_object:
        label, distance = closest_object
        current_time = time.time()
        
        if last_distance is None:
            last_distance = distance
            last_object = label

        if last_object != label or abs(last_distance - distance) > 2:  # threshold to avoid repeated messages
            if current_time - last_speak_time > speak_interval:
                direction = "nearing" if distance < last_distance else "moving away"
                message = f"A {label} is {distance:.2f} inches away and is {direction}"
                message_queue.append(message)
                last_speak_time = current_time
                last_distance = distance
                last_object = label

    # Display the frame
    cv2.imshow('Webcam Image', frame)

    # Add a break condition (Press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
camera.release()
cv2.destroyAllWindows()
