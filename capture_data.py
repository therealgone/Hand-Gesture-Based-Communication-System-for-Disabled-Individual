import cv2
import numpy as np
import mediapipe as mp
import os
import json
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Create dataset directory if it doesn't exist
DATASET_DIR = "hand_dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Dictionary to store landmarks
landmarks_data = {}

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        full = []
        for i in range(2):
            if i < len(result.multi_hand_landmarks):
                full.extend([c for lm in result.multi_hand_landmarks[i].landmark for c in (lm.x, lm.y, lm.z)])
            else:
                full.extend([0.0] * 63)
        return np.array(full)
    return None

def save_landmarks(label, landmarks):
    if label not in landmarks_data:
        landmarks_data[label] = []
    landmarks_data[label].append(landmarks.tolist())

def save_dataset():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"hand_dataset_{timestamp}.json"
    with open(os.path.join(DATASET_DIR, filename), 'w') as f:
        json.dump(landmarks_data, f)
    print(f"Dataset saved to {filename}")

current_label = ""
collecting = False
frame_count = 0
samples_per_label = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Display instructions
    cv2.putText(frame, "Press '1-9' to select label", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'c' to start/stop collecting", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 's' to save dataset", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if current_label:
        cv2.putText(frame, f"Current label: {current_label}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if collecting:
        cv2.putText(frame, "COLLECTING...", (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Samples collected: {frame_count}/{samples_per_label}", (10, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            save_landmarks(current_label, landmarks)
            frame_count += 1
            
            if frame_count >= samples_per_label:
                collecting = False
                frame_count = 0
                print(f"Collected {samples_per_label} samples for label {current_label}")
    
    cv2.imshow("Hand Gesture Capture", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord(str(i)) for i in range(1, 10)]:
        current_label = f"gesture_{key - ord('0')}"
        print(f"Selected label: {current_label}")
    elif key == ord('c'):
        if current_label:
            collecting = not collecting
            if collecting:
                print(f"Started collecting for label {current_label}")
            else:
                print("Stopped collecting")
    elif key == ord('s'):
        save_dataset()
        print("Dataset saved!")

cap.release()
cv2.destroyAllWindows() 