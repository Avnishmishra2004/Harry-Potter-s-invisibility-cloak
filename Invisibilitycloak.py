import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)
time.sleep(2)  # Allowing the camera to adjust

# Capture the background frame
background = None
for i in range(30):  # Capture multiple frames for a stable background
    ret, background = cap.read()
background = cv2.flip(background, 1)  # Flip to match the live frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for a natural mirror effect

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for detecting red cloak
    lower_red1 = np.array([0, 120, 70])   # First lower range of red
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])  # Second lower range of red
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red detection
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine both masks
    mask = mask1 + mask2

    # Refine the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Create an inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Extract the background for the masked region
    background_part = cv2.bitwise_and(background, background, mask=mask)

    # Extract the foreground (non-cloak area)
    foreground_part = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine both parts to get the final frame
    final_output = cv2.addWeighted(background_part, 1, foreground_part, 1, 0)

    # Display the output
    cv2.imshow("Invisibility Cloak (Red)", final_output)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
