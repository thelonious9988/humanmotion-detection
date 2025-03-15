import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pygame

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Load YOLOv8 modelS
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Motion detection settings
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
motion_threshold = 15000

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # Convert frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Background subtraction
    fgmask = fgbg.apply(frame)
    motion_score = np.sum(fgmask)

    # Draw bounding boxes for detected humans
    human_detected = False
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if classes[i] == 0:  # Class 0 = "person"
            human_detected = True

    # Trigger alarm only if motion and a human are detected
    if motion_score > motion_threshold and human_detected:
        print("Human detected! Triggering alarm...")
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(r'D:\warning_tone.mp3')
            pygame.mixer.music.play()

    # Display the frame
    cv2.imshow("Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()