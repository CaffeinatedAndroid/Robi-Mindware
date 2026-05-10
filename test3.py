import cv2
import numpy as np
# 1. Load image and find contours
image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 2. Loop through contours to get coordinates
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    print(f"Top-Left: ({x}, {y}), Bottom-Right: ({x + w}, {y + h})")

for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print(f"Center: ({cX}, {cY})")

rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)
box = np.int32(box)  # Preferred for OpenCV compatibility
cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
print(f"Corners: {box}")
cv2.imshow('Output', image)
cv2.imwrite('output.png', image)
