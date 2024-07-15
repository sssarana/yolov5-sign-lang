import torch

# Weights
custom_weights = "C:\\Users\\Sofiia\\yolov5\\best.pt"

# Model
model = torch.hub.load('C:\\Users\\Sofiia\\yolov5', 'custom', 'best.pt', source='local')  # or yolov5n - yolov5x6, custom path=custom_weights

# Images
img = "C:\\Users\\Sofiia\\yolov5\\U_test.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()
results.show()