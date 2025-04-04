import torch
import cv2
from ultralytics import YOLO

# Load the trained YOLOv11 model
model_path = "weights/best.pt"  # Ensure this path is correct
model = YOLO(model_path)

# Load an image for testing
image_path = "IMAGES/onlyknife.jpg"  # Change this to your test image path
image = cv2.imread(image_path)

# Perform inference
results = model(image)

# Display results
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0])  # Class label
        label = f"{model.names[cls]}: {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image
#cv2.imshow("Detection", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("output.jpg", image)
print("Output saved as output.jpg")

