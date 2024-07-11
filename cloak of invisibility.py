
import cv2
import numpy as np
import time

# Capture the video
cap = cv2.VideoCapture(0)

# Allow the camera to warm up
time.sleep(2)

# Capture the background frame
for i in range(30):
    ret, background = cap.read()

# Flip the background frame
background = np.flip(background, axis=1)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame
    frame = np.flip(frame, axis=1)
    
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the blue color range in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    # Create a mask to detect blue color
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Refine the mask
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    
    # Create an inverted mask
    mask2 = cv2.bitwise_not(mask1)
    
    # Segment the blue color part out of the frame using mask1
    res1 = cv2.bitwise_and(frame, frame, mask=mask2)
    
    # Replace the blue color part with the background using mask2
    res2 = cv2.bitwise_and(background, background, mask=mask1)
    
    # Combine both the results
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    
    # Display the output
    cv2.imshow('Invisibility Cloak', final_output)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
