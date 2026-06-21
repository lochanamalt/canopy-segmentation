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

# Reconstruct the Model
model = smp.DeepLabV3Plus(
    encoder_name="mobilenet_v2",
    encoder_weights=None,  # No need for imagenet weights
    in_channels=3,
    classes=1
).to(DEVICE)

model.load_state_dict(torch.load("models/best_canopy_seg_model.pth", map_location=DEVICE))
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
        canopy_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    binary_mask = (canopy_mask > 0.5).astype(np.uint8)
    overlay = original_image.copy()
    overlay[canopy_mask > 0.5] = [0, 255, 0]  # Green color
    combined_with_img = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)

    return binary_mask, combined_with_img


# Usage
mask, combined = predict_canopy("test_image/date_24-5-2025_10.0.10_1.png")
cv2.imwrite("test_image/output_mask.png", mask * 255)  # Multiply by 255 to make it visible
cv2.imwrite("test_image/overlay.png", combined)

