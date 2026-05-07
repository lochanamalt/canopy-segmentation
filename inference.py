"""
@author: Lochana Marasinghe
@date: 5/6/2026
@description: 
"""
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Reconstruct the Model
model = smp.DeepLabV3Plus(
    encoder_name="mobilenet_v2",
    encoder_weights=None,  # No need for imagenet weights since we are loading yours
    in_channels=3,
    classes=1
).to(DEVICE)

model.load_state_dict(torch.load("best_canopy_seg_model.pth", map_location=DEVICE))
model.eval()

transform = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])


def predict_canopy(image_path):

    original_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        # Apply Sigmoid to get probabilities (0.0 to 1.0)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()

    binary_mask = (mask > 0.5).astype(np.uint8)
    overlay = original_image.copy()
    overlay[mask > 0.5] = [0, 255, 0]  # Green color
    combined = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)

    return binary_mask, combined


# Usage
mask, combined = predict_canopy("yolo_annotations/test/images/pi2_date_18-6-2025_15-0-10_1_png.rf.1c089affd68a1972231bdd3e3b8120cc.jpg")
cv2.imwrite("output_mask.png", mask * 255)  # Multiply by 255 to make it visible
cv2.imwrite("overlay.png", combined)
# Create a green tint for the detected canopy
