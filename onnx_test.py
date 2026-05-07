"""
@author: Lochana Marasinghe
@date: 5/6/2026
@description: 
"""
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from mask_dataset import WheatBinaryDataset

BATCH_SIZE = 16

# Training Config
train_transform = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])

# 1. Load both models
orig_session = ort.InferenceSession("deeplab_canopy_seg.onnx")
quant_session = ort.InferenceSession("deeplab_canopy_seg_int8.onnx")

test_dataset = WheatBinaryDataset('masks/test', transform=train_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

# 2. Prepare an input image (using one batch from your loader)
images, _ = next(iter(test_loader))
input_data = images.numpy().astype(np.float32)

# 3. Run inference
input_name = orig_session.get_inputs()[0].name
orig_pred = orig_session.run(None, {input_name: input_data})[0]
quant_pred = quant_session.run(None, {input_name: input_data})[0]

# 4. Post-process (DeepLab usually outputs [Batch, Classes, H, W])
# Get the mask by taking the class with the highest probability
orig_mask = np.argmax(orig_pred[0], axis=0)
quant_mask = np.argmax(quant_pred[0], axis=0)

# 5. Visualize side-by-side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# Show original image (re-normalize if needed)
axes[0].imshow(input_data[0].transpose(1, 2, 0))
axes[0].set_title("Original Image")

axes[1].imshow(orig_mask)
axes[1].set_title("Original Float32 Mask")

axes[2].imshow(quant_mask)
axes[2].set_title("Quantized Int8 Mask")

plt.show()
