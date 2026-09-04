"""
@author: Lochana Marasinghe
@date: 9/4/2026
@description: 
"""
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import onnxruntime as ort

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
MODEL_PATH = "models/deeplab_canopy_seg_int8.onnx"
session = ort.InferenceSession(MODEL_PATH, options, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

transform = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])

def run_canopy_inference(image_path: str):

    original_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    tensor_img = transform(image=image_rgb)["image"]
    input_numpy = tensor_img.unsqueeze(0).numpy().astype(np.float32)


    # Run Inference
    outputs = session.run([output_name], {input_name: input_numpy})
    mask_logits = outputs[0]  # Shape: [1, 1, 628, 928]

    # Apply Sigmoid to get probabilities (0.0 to 1.0)
    probabilities = 1 / (1 + np.exp(-mask_logits))
    binary_mask = (probabilities[0, 0] > 0.5).astype(np.uint8) * 255

    overlay = original_image.copy()
    overlay[probabilities[0, 0] > 0.5] = [0, 255, 0]  # Green color
    combined_with_img = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)

    return binary_mask, combined_with_img

mask, combined = run_canopy_inference("test_image/pi1_date_20-5-2024_15.0.11_1.png")
cv2.imwrite("test_image/int8_onnx_pi1_date_20-5-2024_15.0.11_1_output_mask.png", mask )
cv2.imwrite("test_image/int8_onnx_pi1_date_20-5-2024_15.0.11_1_overlay.png", combined)


