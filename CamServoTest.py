import cv2
import numpy as np
import time

# Initialize video capture and servo library (e.g., pigpio or PCA9685)
 cap = cv2.VideoCapture(0)
# servo = initialize_servo()

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Detect object (Example: Red color mask)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # 2. Calculate center
        obj_center_x = x + w / 2
        obj_center_y = y + h / 2
        frame_center_x = frame.shape[1] / 2
        frame_center_y = frame.shape[0] / 2

        # 3. Calculate errors
        error_x = frame_center_x - obj_center_x
        error_y = frame_center_y - obj_center_y

        # 4. Move servos based on error
        # Example: Simple proportional step
        if abs(error_x) > 10:
            servo.pan(error_x) # Function adjusts angle based on sign/magnitude
        if abs(error_y) > 10:
            servo.tilt(error_y)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
