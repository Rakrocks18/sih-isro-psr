from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

results = model("../trials_images/meowwwwwwww.jpeg")
for result in results:
    result.save(filename="seg_img.jpg")
