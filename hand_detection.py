import cv2 #Used for capturing and processing video frames.
import mediapipe as mp #Provides the pre-trained Hand Tracking model for detecting and drawing hand landmarks.


mp_hands = mp.solutions.hands #Access the MediaPipe Hands module
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)  #Creates a hand tracking model with these parameters:
mp_drawing = mp.solutions.drawing_utils # Used to draw hand landmarks on frames.

cap = cv2.VideoCapture(0) #Opens the default camera (0 refers to the first camera).

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(image_rgb)

    # Draw hand landmarks on the frame for both hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Dual-Hand Gesture Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
