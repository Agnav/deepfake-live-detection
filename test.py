import cv2
import dlib
import numpy as np
import pyvirtualcam  # For virtual webcam
from faceswap_cam import face_swap
from detection import Face_Detector, Landmark_Detector

# Initialize face detection and swapping modules
face_detector = Face_Detector()
lmk_detector = Landmark_Detector()
detector = dlib.get_frontal_face_detector()
FS = face_swap("madhu.jpg")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the webcam feed resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Start the virtual camera
with pyvirtualcam.Camera(width=400, height=300, fps=30) as cam:
    print(f"Using virtual camera: {cam.device}")
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (400, 300))
        swapped_face = frame.copy()  # Fallback to original frame

        # Detect faces
        bboxes, _ = face_detector.detect(frame)
        if len(bboxes) > 0:
            # Process the first detected face
            bbox = bboxes[0].astype(np.int_)
            lmks, PRY_3d = lmk_detector.detect(frame, bbox)
            lmks = lmks.astype(np.int_)
            try:
                swapped_face = FS.run(frame, lmks)
            except Exception as e:
                print(f"Face swap error: {e}")
        else:
            print("No face detected. Showing original frame.")

        # Send the swapped face to the virtual webcam
        cam.send(swapped_face)
        cam.sleep_until_next_frame()

        # Display the swapped face (optional for debugging)
        # cv2.imshow('Face Swapper', swapped_face)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
