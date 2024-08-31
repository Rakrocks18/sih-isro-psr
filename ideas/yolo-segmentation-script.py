import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('yolov8n-seg.pt')  # or use your custom trained model

def segment_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform segmentation
    results = model(image)

    # Get the segmentation masks
    masks = results[0].masks.data.cpu().numpy()

    # Create a blank mask to store all segmentations
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Combine all masks
    for mask in masks:
        full_mask = np.logical_or(full_mask, mask).astype(np.uint8) * 255

    return image, full_mask

def save_segmented_pixels(image, mask, output_path):
    # Apply the mask to the original image
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Save the segmented image
    plt.imsave(output_path, segmented_image)

# Example usage
image_path = 'path/to/your/image.jpg'
output_path = 'path/to/save/segmented_image.png'

image, mask = segment_image(image_path)
save_segmented_pixels(image, mask, output_path)

print(f"Segmented image saved to {output_path}")
