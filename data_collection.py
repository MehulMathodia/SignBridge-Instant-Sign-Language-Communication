import cv2
import mediapipe as mp
import os
import numpy as np

# Create a directory to store the data
DATA_DIR = 'gesture_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of gestures and samples per gesture
NUM_GESTURES = 36  # 0-9 (10) + A-Z (26) = 36 gestures
SAMPLES_PER_GESTURE = 100  # Number of samples to collect for each gesture

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Define gesture labels (0-9, A-Z)
gesture_labels = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]

for gesture_id, gesture_label in enumerate(gesture_labels):
    gesture_dir = os.path.join(DATA_DIR, gesture_label)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    print(f'Collecting data for gesture {gesture_label} (ID: {gesture_id})...')

    # Wait for the user to be ready
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Press "s" for gesture {gesture_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Collect SAMPLES_PER_GESTURE samples
    sample_count = 0
    while sample_count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Save the landmarks as a numpy array
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                np.save(os.path.join(gesture_dir, f'{sample_count}.npy'), landmarks)
                sample_count += 1

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
