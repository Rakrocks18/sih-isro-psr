from ultralytics import YOLO
import random
import cv2
import numpy as np

model = YOLO("yolov8n-seg.pt")

results = model("/content/meowwwwwwww.jpeg")
for result in results:
    result.save(filename="seg_img.jpg")