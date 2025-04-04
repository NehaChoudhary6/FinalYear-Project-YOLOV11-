# import time
# from ultralytics import YOLO
#
# model = YOLO("yolo11m.pt")  # Change model name accordingly
#
# image = "IMAGES/knife-hai.jpg"
#
# start_time = time.time()
# results = model(image)
# end_time = time.time()
#
# inference_time = end_time - start_time
# fps = 1 / inference_time
#
# print(f"Inference Time: {inference_time:.3f} seconds")
# print(f"FPS: {fps:.2f}")
import os

model_path = "yolo11m.pt"  # Change for different models
size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB

print(f"Model Size: {size:.2f} MB")
