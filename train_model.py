import numpy as np
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_dataset(dataset_dir):
    all_data = []
    all_labels = []
    
    # Load all JSON files in the dataset directory
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.json'):
            with open(os.path.join(dataset_dir, filename), 'r') as f:
                data = json.load(f)
                for label, landmarks_list in data.items():
                    for landmarks in landmarks_list:
                        all_data.append(landmarks)
                        all_labels.append(label)
    
    return np.array(all_data), np.array(all_labels)

def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model():
    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset("hand_dataset")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label mapping
    np.save("hand_labels_final.npy", label_encoder.classes_)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Normalize data
    X_train = (X_train - np.mean(X_train)) / (np.std(X_train) + 1e-6)
    X_test = (X_test - np.mean(X_test)) / (np.std(X_test) + 1e-6)
    
    # Create and train model
    print("Creating model...")
    model = create_model(X_train.shape[1], len(label_encoder.classes_))
    
    # Callbacks
    checkpoint = ModelCheckpoint("best_hand_model.h5", 
                               monitor='val_accuracy',
                               save_best_only=True,
                               mode='max',
                               verbose=1)
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                 patience=10,
                                 restore_best_weights=True)
    
    # Train model
    print("Training model...")
    history = model.fit(X_train, y_train,
                       epochs=50,
                       batch_size=32,
                       validation_data=(X_test, y_test),
                       callbacks=[checkpoint, early_stopping])
    
    # Save final model
    model.save("hand_classifier_final.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    train_model() 