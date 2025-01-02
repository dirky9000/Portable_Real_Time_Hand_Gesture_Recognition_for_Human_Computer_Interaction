import os
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define input folder and classes
input_folder = "/home/dirky9000/Desktop/hand"
output_folder = "/home/dirky9000/Desktop/output_images"
os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
classes = ["thumb_up", "thumb_down", "open_hand", "horns", "peace"]

# Process one image per class
for class_name in classes:
    class_folder = os.path.join(input_folder, class_name)
    print(f"Processing class: {class_name}")

    # Process the first valid image in the folder
    for filename in os.listdir(class_folder):
        if filename.endswith((".JPG", ".jpeg", ".png")):  # Accept JPEG and PNG images
            filepath = os.path.join(class_folder, filename)
            print(f"Processing image: {filepath}")

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
                # Save original image
                original_image_path = os.path.join(output_folder, f"{class_name}_original.png")
                cv2.imwrite(original_image_path, image)

                # Save image with landmarks
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                annotated_image_path = os.path.join(output_folder, f"{class_name}_landmarks.png")
                cv2.imwrite(annotated_image_path, annotated_image)

                print(f"Saved images for class: {class_name}")
                break  # Process only one image per class
            else:
                print(f"No hand landmarks detected in {filepath}")
                break
