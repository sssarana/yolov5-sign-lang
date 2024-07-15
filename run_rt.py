import torch
import cv2

# Weights
custom_weights = "C:\\Users\\Sofiia\\yolov5\\best.pt"

# Model
model = torch.hub.load('C:\\Users\\Sofiia\\yolov5', 'custom', 'best.pt', source='local')

# Video stream from TCP server
video_stream_url = "tcp://192.168.137.125:34888"
cap = cv2.VideoCapture(video_stream_url) # replace video_stream_url with 0 to test on local machine

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Inference
    results = model(frame)

    # Get image with results
    annotated_frame = results.render()[0]

    # Show the frame with detection boxes
    cv2.imshow('Real-time Detection', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
