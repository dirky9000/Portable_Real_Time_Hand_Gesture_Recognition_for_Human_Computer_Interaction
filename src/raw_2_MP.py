import os
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define input and output folders
input_folder = "/home/dirky9000/Desktop/hand"  # Path to the folder containing gesture subfolders
output_folder = "/home/dirky9000/Desktop/meep"
os.makedirs(output_folder, exist_ok=True)

# Define the class folders (5 gesture folders)
classes = ["thumb_up", "thumb_down", "open_hand", "horns", "peace"]

# Iterate through each class folder
for class_name in classes:
    class_folder = os.path.join(input_folder, class_name)
    if not os.path.exists(class_folder):
        print(f"Folder not found: {class_folder}")
        continue

    for filename in os.listdir(class_folder):
        if filename.endswith(('.JPG', '.jpeg', '.png', '.jpg')):  # Accept JPEG and PNG files
            filepath = os.path.join(class_folder, filename)
            print(f"Processing image: {filepath}")

            # Read and resize the image
            image = cv2.imread(filepath)
            if image is None:
                print(f"Error reading image: {filepath}")
                continue

            image = cv2.resize(image, (640, 480))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the image
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # Save the processed image
            output_path = os.path.join(
                output_folder, f"{class_name}_processed_{filename}"
            )
            cv2.imwrite(output_path, image)
            print(f"Saved processed image: {output_path}")

print(f"All images processed and saved to {output_folder}.")

# Release MediaPipe resources
hands.close()
