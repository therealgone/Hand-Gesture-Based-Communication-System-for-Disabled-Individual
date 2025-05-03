import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import google.generativeai as genai
from tensorflow.keras.models import load_model
import time
import os
import json
from dotenv import load_dotenv
import requests

# ========== Load Gemini API Key ==========
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

# ========== Load Models ==========
print("\n=== Loading Pose Model ===")
pose_model = load_model("pose_classifier_final_new.h5")
pose_labels = np.load("pose_labels_final_new.npy")
print(f"Pose model loaded with {len(pose_labels)} classes")
print("Pose labels:", pose_labels)

print("\n=== Loading Gesture Model ===")
gesture_model = load_model("preset_gesture_model.h5")
gesture_labels = np.load("preset_gesture_labels.npy")
print(f"Gesture model loaded with {len(gesture_labels)} classes")
print("Gesture labels:", gesture_labels)

# ========== Load Presets from JSON file ==========
PRESETS_FILE = "presets.json"
PRESET_UPDATE_FLAG = "preset_update.flag"
last_preset_check = time.time()

def load_presets():
    if not os.path.exists(PRESETS_FILE):
        # Create default presets that match your gesture_labels
        presets = {}
        for i, label in enumerate(gesture_labels):
            if label.startswith("preset"):
                presets[label] = label  # Default value same as key, will be customized by user
        
        # If no presets found in labels, create defaults
        if not presets:
            presets = {
                "preset1": "hello",
                "preset2": "thank you",
                "preset3": "help me"
            }
        
        with open(PRESETS_FILE, 'w') as f:
            json.dump(presets, f)
    else:
        with open(PRESETS_FILE, 'r') as f:
            presets = json.load(f)
    
    return presets

presets = load_presets()

# ========== Text-to-Speech ==========
tts = pyttsx3.init()
tts.setProperty('rate', 160)

# ========== Mediapipe Setup ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Try to set higher FPS

last_prediction = ""
cooldown_counter = 0
cooldown_frames = 30  # Reduced from 60 to improve speed
detection_delay = 10  # Reduced from 20

# Set up Web Server URL
WEB_SERVER_URL = "http://localhost:5000"

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

# Variables for detection stability
detection_history = []
HISTORY_SIZE = 5
STABILITY_THRESHOLD = 0.7

# Variables to check if prediction has been sent
last_sent_prediction = ""
frame_counter = 0

# Dict to map from preset label to user-defined value
def update_preset_mapping():
    global preset_mapping
    preset_mapping = {}
    for preset_key, preset_value in presets.items():
        # Find the index of this preset in gesture_labels
        if preset_key in gesture_labels:
            preset_mapping[preset_key] = preset_value
    print("Updated preset mappings:", preset_mapping)

update_preset_mapping()
print("Available gesture labels:", gesture_labels)

while True:
    # Check if presets have been updated (every 2 seconds)
    current_time = time.time()
    if current_time - last_preset_check > 2:
        last_preset_check = current_time
        
        # Check if the update flag file exists
        if os.path.exists(PRESET_UPDATE_FLAG):
            try:
                # Load updated presets
                presets = load_presets()
                update_preset_mapping()
                print("Presets reloaded from file:", presets)
                
                # Remove the flag file
                os.remove(PRESET_UPDATE_FLAG)
            except Exception as e:
                print(f"Error reloading presets: {e}")
    
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_counter += 1

    # Only process every other frame to improve speed
    if frame_counter % 2 != 0:
        cv2.imshow("Sign Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    landmark = extract_landmarks(frame)
    if landmark is not None:
        norm = (landmark - np.mean(landmark)) / (np.std(landmark) + 1e-6)

        # Check for regular words with pose model
        pose_pred = pose_model.predict(np.expand_dims(norm, axis=0), verbose=0)[0]
        pose_word_idx = np.argmax(pose_pred)
        pose_word = pose_labels[pose_word_idx]
        pose_conf = np.max(pose_pred)

        # Also check for preset gestures
        gesture_pred = gesture_model.predict(np.expand_dims(norm, axis=0), verbose=0)[0]
        gesture_idx = np.argmax(gesture_pred)
        gesture_label = gesture_labels[gesture_idx]
        gesture_conf = np.max(gesture_pred)

        # Decide which prediction to use based on confidence
        word = ""
        if gesture_conf > 1.0 and gesture_label in preset_mapping:
            # Use the preset mapping to get the user-defined value
            word = preset_mapping[gesture_label]
            cv2.putText(frame, f"Preset: {gesture_label} -> {word}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif pose_conf > 0.90:
            word = pose_word

        # Add to history for stability if we have a word
        if word:
            detection_history.append(word)
            if len(detection_history) > HISTORY_SIZE:
                detection_history.pop(0)
            
            # Check if we have a stable prediction
            if len(detection_history) == HISTORY_SIZE:
                most_common = max(set(detection_history), key=detection_history.count)
                frequency = detection_history.count(most_common) / HISTORY_SIZE
                
                if frequency >= STABILITY_THRESHOLD and most_common != last_sent_prediction:
                    last_prediction = most_common
                    last_sent_prediction = most_common
                    # Update the web interface
                    try:
                        requests.post(f"{WEB_SERVER_URL}/update_word", json={"word": last_prediction}, timeout=0.5)
                    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
                        pass  # Don't wait for the response to avoid slowing down
                    
                    # Display the detected word
                    cv2.putText(frame, f"Detected: {last_prediction}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()