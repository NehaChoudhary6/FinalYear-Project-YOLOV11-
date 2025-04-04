import torch
import cv2
import mediapipe as mp
import ssl
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Disable SSL certificate verification (to handle SSL errors)
ssl._create_default_https_context = ssl._create_unverified_context

# Load the YOLOv11n model for weapon detection
model = YOLO("yolo11n.pt")  # Ensure you have 'yolo11n.pt' trained for weapons

# Print class names for debugging
print("Model class names:", model.names)  # Debug model classes

# Define weapon classes (adjust based on actual model training)
weapon_classes = ["knife", "Pistol" , "Rifle"]

# Load the image
img_path = 'IMAGES/knife7.jpg'  # Change this to the path of your image
img = cv2.imread(img_path)

# Perform weapon detection using YOLOv11n
results = model(img_path)

# Extract and print the weapon names detected
for box in results[0].boxes:  # Accessing the first image's detection results
    conf = box.conf.item()  # Confidence score
    cls = int(box.cls.item())  # Class ID
    weapon_name = model.names[cls]

    # Only print detected weapons
    if weapon_name in weapon_classes and conf > 0.3:
        print(f"Detected weapon: {weapon_name} with confidence: {conf:.2f}")

# Overlay detections on the image
annotated_image = results[0].plot()

# Save the output image instead of showing it (fixes OpenCV error)
cv2.imwrite("output.jpg", annotated_image)
print("Output image saved as output.jpg")

# Display image using matplotlib (if GUI is available)
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
