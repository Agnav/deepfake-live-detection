import cv2

# Try different device indices to find the correct virtual camera
for i in range(10):  # Test the first 10 possible camera indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Device {i} is accessible.")
        cap.release()
    else:
        print(f"Device {i} is not accessible.")
