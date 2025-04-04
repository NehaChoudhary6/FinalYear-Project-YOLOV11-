from ultralytics import YOLO

# Load your trained model
model = YOLO("weights/best.pt")  # Make sure 'best.pt' is in the correct path

# Update dataset path (local path, not Colab)
dataset_path =r"C:\Users\User\PycharmProjects\FINALYEAR-project\DATASET\data.yaml"

# Run evaluation
metrics = model.val(data=dataset_path)

# Print results
print(metrics)
