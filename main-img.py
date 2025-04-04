import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 🔹 Load the trained YOLOv11 model
model = YOLO("weights/best.pt")  # Change to "yolo11n.pt" if needed

# 🔹 Define your weapon classes (update according to your training)
weapon_classes = ["knife", "Pistol", "Rifle"]

# 🔹 Load the image
image_path = "IMAGES/police-pistol.jpg"  # Change to your image path
image = cv2.imread(image_path)

# 🔹 Perform detection
results = model(image)

# 🔹 Initialize list to store detected weapons
detected_weapons = []

# 🔹 Process results
for result in results:
    for box in result.boxes:
        conf = box.conf.item()
        cls = int(box.cls.item())
        weapon_name = model.names[cls]

        if weapon_name in weapon_classes and conf > 0.3:
            detected_weapons.append(f"{weapon_name} ({conf:.2f})")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{weapon_name}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 🔹 Print detected weapon names
if detected_weapons:
    print(" Detected Weapons:")
    for weapon in detected_weapons:
        print(f"✅ {weapon}")
else:
    print("❌ No weapons detected.")

# 🔹 Save and show the result image
cv2.imwrite("output.jpg", image)
print("Output image saved as 'output.jpg'")

# 🔹 Show with matplotlib (for better rendering)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Weapon Detection Output")
plt.show()
