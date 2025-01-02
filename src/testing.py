from picamera2 import Picamera2
import cv2
import mediapipe as mp
import joblib
import numpy as np
from gpiozero import LED
import time
from libcamera import Transform

# Load the trained SVM model
model_filename = "gesture_recognition_svm.pkl"
svm = joblib.load(model_filename)

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}, transform=Transform(vflip=True)))
picam2.start()
time.sleep(2)  # Allow the camera to warm up

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize GPIO LEDs
red_led = LED(22)   # Red LED
green_led = LED(27) # Green LED
blue_led = LED(17)  # Blue LED

# Class labels (adjust based on your training labels)
classes = ["thumbs_up", "thumbs_down", "open_hand", "horns", "peace"]

# Initialize FPS calculation variables
start_time = time.time()
frame_count = 0
fps = 0.0

# For calculating the average FPS
total_frames = 0
total_time = 0.0

print("Starting real-time gesture recognition with LED control. Press 'q' to exit.")

try:
    while True:
        # Capture frame from Picamera2
        frame = picam2.capture_array()
        frame_count += 1  # Increment frame count for FPS calculation
        total_frames += 1  # Increment total frame count

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        # Check for hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks into a flat list
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Convert landmarks to numpy array for SVM prediction
                landmarks_np = np.array(landmarks).reshape(1, -1)

                # Predict the gesture
                prediction = svm.predict(landmarks_np)[0]
                gesture = classes[prediction]
                # Control LEDs based on the gesture
                if gesture == "thumbs_up":
                    red_led.off()
                    green_led.off()
                    blue_led.on()
                elif gesture == "thumbs_down":
                    red_led.on()
                    green_led.off()
                    blue_led.off()
                elif gesture == "open_hand":
                    red_led.off()
                    green_led.on()
                    blue_led.off()
                elif gesture == "horns":
                    red_led.on()
                    green_led.off()
                    blue_led.on()
                elif gesture == "peace":
                    red_led.on()
                    green_led.on()
                    blue_led.off()
                else:
                    # Turn off all LEDs for unknown gestures
                    red_led.off()
                    green_led.off()
                    blue_led.off()

                # Display recognized gesture on the frame
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)

        # Calculate FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:  # Update every second
            fps = frame_count / elapsed_time
            total_time += elapsed_time  # Add elapsed time to total time
            frame_count = 0
            start_time = time.time()

        # Overlay FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)

        # Display the video feed with overlays
        cv2.imshow("Gesture Recognition", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Calculate and print average FPS
    if total_time > 0:
        average_fps = total_frames / total_time
        print(f"\nAverage FPS: {average_fps:.2f}")
    else:
        print("No frames processed. Cannot calculate FPS.")

    # Clean up resources
    picam2.stop()
    hands.close()
    cv2.destroyAllWindows()
    red_led.off()
    green_led.off()
    blue_led.off()