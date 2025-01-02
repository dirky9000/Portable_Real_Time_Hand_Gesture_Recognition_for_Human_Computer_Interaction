import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define input folder and classes
input_folder = "/home/dirky9000/Desktop/hand"
output_csv = "hand_gesture_landmarks.csv"
classes = ["thumb_up", "thumb_down", "open_hand", "horns", "peace"]

data = []  # To store landmarks and labels

# Iterate over each class folder
for label, class_name in enumerate(classes):
    class_folder = os.path.join(input_folder, class_name)
    print(f"Processing class: {class_name}")
    for filename in os.listdir(class_folder):
        if filename.endswith((".JPG", ".jpeg", ".png")):  # Accept JPEG and PNG images
            filepath = os.path.join(class_folder, filename)
            print(f"Processing image: {filepath}")  # Show progress for each image

            # Read and resize image
            image = cv2.imread(filepath)
            if image is None:
                print(f"Error reading image: {filepath}")
                continue
            image = cv2.resize(image, (640, 480))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])  # Append x, y, z coordinates
                    landmarks.append(label)  # Add class label
                    data.append(landmarks)  # Add to dataset
            else:
                print(f"No hand landmarks detected in {filepath}")

# Save data to CSV
if data:  # Check if data is not empty
    columns = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Landmarks saved to {output_csv}")
else:
    print("No landmark data was saved. Please check your images and setup.")