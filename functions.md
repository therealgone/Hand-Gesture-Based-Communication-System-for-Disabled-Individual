# Function Documentation

## Main Application Functions

### test.py

#### `extract_landmarks(frame)`
- **Purpose**: Extracts hand landmarks from a video frame
- **Parameters**:
  - `frame`: OpenCV video frame
- **Returns**: Normalized landmark array or None if no hands detected
- **Description**: Uses MediaPipe to detect hand landmarks and normalizes them for model input

#### `update_preset_mapping()`
- **Purpose**: Updates the mapping between preset gesture labels and user-defined values
- **Description**: Reads presets.json and creates a mapping dictionary for gesture recognition

### bluetooth_webserver.py

#### `load_presets()`
- **Purpose**: Loads or creates preset gesture mappings
- **Returns**: Dictionary of preset mappings
- **Description**: Reads presets.json or creates default presets if file doesn't exist

#### `update_prediction()`
- **Purpose**: Updates the current prediction on the web interface
- **Parameters**:
  - `prediction`: The detected sign language word
- **Description**: Sends the prediction to the web interface via POST request

## Model Functions

### Pose Model
- **File**: `pose_classifier_final_new.h5`
- **Input**: 126-dimensional normalized landmark array
- **Output**: 39 classes of sign language words
- **Architecture**: 
  - Dense(256) → BatchNormalization → Dropout(0.2)
  - Dense(128) → Dropout(0.2)
  - Dense(64)
  - Dense(39, softmax)

### Gesture Model
- **File**: `preset_gesture_model.h5`
- **Input**: 126-dimensional normalized landmark array
- **Output**: 5 classes (3 presets + 'time' + 'what')
- **Architecture**:
  - Dense(256) → Dropout
  - Dense(128) → Dropout
  - Dense(5, softmax)

## Data Processing

### Landmark Normalization
- **Purpose**: Normalizes landmark coordinates for consistent model input
- **Process**:
  1. Extract x, y, z coordinates
  2. Calculate center point
  3. Scale coordinates relative to center
  4. Normalize using mean and standard deviation

### Prediction Stability
- **Purpose**: Ensures stable predictions by using a history buffer
- **Parameters**:
  - `HISTORY_SIZE`: 5 (number of previous predictions to consider)
  - `STABILITY_THRESHOLD`: 0.7 (minimum frequency for stable prediction)
- **Process**:
  1. Maintain a history of recent predictions
  2. Count frequency of each prediction
  3. Only accept predictions that meet the stability threshold

## Web Interface

### Routes
- `/`: Main interface page
- `/update_word`: Endpoint for receiving new predictions
- `/update_presets`: Endpoint for updating preset configurations

### Features
- Real-time video display
- Current prediction display
- Preset configuration interface
- Bluetooth device status 