#!/usr/bin/env python
# coding: utf-8

# Pinnacle final code draft

# I have used Ultralytics (YOLOv8 model) and OpenCV
# Installing the required dependencies:
'''get_ipython().system('pip install opencv-python')
get_ipython().system('pip install ultralytics')'''


# Importing Required Libraries
# cv2: It is the OpenCV python library
# imported the YOLO from ultralytics to load the model and work upon it

import os            # interacts with your operating system (deletes the existing file)
import cv2           
from ultralytics import YOLO      
import numpy as np
import time
from gtts import gTTS  # Text-to-Speech
import pygame
from threading import Thread     # Runs tasks (like speech synthesis) concurrently with object detection.



# Initialize Text-to-Speech, allowing audio playback later in the code.
pygame.mixer.init()



''' Function below converts text into speech and plays it. It is used to announce detected objects. '''

def speech(text):
    ''' Generate and play speech from text '''
    try:
        print(f"Speaking: {text}")
        language = "en"
        
        # Create a unique filename for each audio file
        file_path = "output_temp.mp3"   # defines a temporary file name for the audio (works like a flag variable, or when we initialize a variable to 0. (i = 0))
        
        # Generate the speech file
        output = gTTS(text=text, lang=language, slow=False)
        output.save(file_path)
        
        # Initialize pygame mixer and play the audio
        # Loads and plays the generated audio file.
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Stop playback and cleanup
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        
        # Remove the audio file after playback
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error in speech synthesis: {e}")

# Function to Generate Class Colors
def get_random_color(cls_num):
    np.random.seed(cls_num)
    return tuple(np.random.randint(0, 256, size=3).tolist())

# Initialize YOLOv8 Model
yolo = YOLO('yolov8s.pt')

# Open Video Capture
video_cap = cv2.VideoCapture(0)
if not video_cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize Variables for Object Detection
font = cv2.FONT_HERSHEY_SIMPLEX
detected_objects = set()  # To store unique detected objects

def detect_and_announce():
    """Main detection loop with real-time announcements."""
    global detected_objects

    prev_time = time.time()  # Initialize prev_time to the current time

    while True:
        ret, frame = video_cap.read()
        if not ret:
            print("Error: Could not read frame from video source.")
            break

        # Perform Object Detection
        results = yolo.predict(frame, stream=True)

        for result in results:
            class_names = result.names  # Get class names

            for box in result.boxes:
                if box.conf[0] > 0.4:  # Confidence threshold
                    # Extract Bounding Box Coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])  # Class index
                    class_name = class_names[cls]  # Class name

                    # Generate Color for the Class
                    color = get_random_color(cls)

                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Add Class Name and Confidence Score
                    label = f"{class_name} {box.conf[0]:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), font, 0.5, color, 2)

                    # Speak new detections
                    if class_name not in detected_objects:
                        detected_objects.add(class_name)
                        Thread(target=speech, args=(f"I found a {class_name}.",)).start()

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        # Display FPS on Frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), font, 0.5, (255, 255, 255), 1)

        # Display the Frame with Detections
        cv2.imshow('Object Detection with YOLOv8', frame)

        # Exit on 'q' or 'ESC' key press
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 'q' or ESC key
            break

    # Release Resources
    video_cap.release()
    cv2.destroyAllWindows()

    # Announce session summary
    if detected_objects:
        summary = ", ".join([f"a {obj}" for obj in detected_objects])
        speech(f"Here are the objects I found: {summary}.")
    else:
        speech("No objects were detected during this session.")

# Start Detection
detect_and_announce()

