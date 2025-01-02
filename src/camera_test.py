from picamera2 import Picamera2
import cv2
import mediapipe as mp
from gpiozero import LED
import time
from libcamera import Transform

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}, transform=Transform(vflip=True)))
picam2.start()
time.sleep(2)  # Allow the camera to warm up

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize GPIO LEDs
red_led = LED(22)
green_led = LED(27) 
blue_led = LED(17)  

# Helper function to recognize gestures
def recognize_gesture(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    if (thumb_tip.y > wrist.y and index_tip.y and middle_tip.y and ring_tip.y and pinky_tip.y): # thumb down
        return "thumb down"
    elif (index_tip.y < thumb_tip.y and
          middle_tip.y < thumb_tip.y and
          ring_tip.y < thumb_tip.y and
          index_tip.y < pinky_tip.y and
          middle_tip.y < pinky_tip.y and
          ring_tip.y < pinky_tip.y and
          thumb_tip.y < wrist.y and
          pinky_tip.y < wrist.y): 
        return "open hand"
    elif (thumb_tip.y < wrist.y and # thumb up
          thumb_tip.y < index_tip.y and 
          thumb_tip.y < middle_tip.y and 
          thumb_tip.y < ring_tip.y and 
          thumb_tip.y < pinky_tip.y and
          index_tip.y < middle_tip.y and
          middle_tip.y < ring_tip.y and
          ring_tip.y < pinky_tip.y):
        return "thumb up"
    elif (pinky_tip.y < ring_tip.y and # horns
          pinky_tip.y < middle_tip.y and 
          pinky_tip.y < index_tip.y and
          thumb_tip.y < ring_tip.y and 
          thumb_tip.y < middle_tip.y and 
          thumb_tip.y < index_tip.y and 
          wrist.y > ring_tip.y and 
          wrist.y > middle_tip.y and 
          wrist.y > index_tip.y): 
        return "horns"
    elif (index_tip.y < thumb_tip.y and
          index_tip.y < ring_tip.y and
          index_tip.y < pinky_tip.y and
          middle_tip.y < thumb_tip.y and
          middle_tip.y < ring_tip.y and
          middle_tip.y < pinky_tip.y and
          wrist.y > pinky_tip.y and
          wrist.y > ring_tip.y and
          wrist.y > thumb_tip.y):
          return "peace"
    else: 
        return "unknown"

# Main loop for gesture recognition
try:
    while True:
        # Capture frame from Picamera2
        frame = picam2.capture_array()

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

                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks)

                # Perform actions based on gesture
                if gesture == "thumb down":
                    red_led.on()
                    green_led.off()
                    blue_led.off()
                elif gesture == "open hand":
                    red_led.off()
                    green_led.on()
                    blue_led.off()
                elif gesture == "thumb up":
                    red_led.off()
                    green_led.off()
                    blue_led.on()
                elif gesture == "horns":
                    red_led.on()
                    green_led.off()
                    blue_led.on()
                elif gesture == "peace":
                    red_led.on()
                    green_led.on()
                    blue_led.off()
                else:
                    red_led.off()
                    green_led.off()
                    blue_led.off()

                # Display recognized gesture
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the video feed with overlays
        cv2.imshow("Gesture Recognition", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Clean up resources
    picam2.stop()
    hands.close()
    cv2.destroyAllWindows()
    red_led.off()
    green_led.off()
    blue_led.off()