# Wheat Canopy Semantic Segmentation using DeepLabV3+

This repository contains a lightweight, semantic segmentation pipeline
designed to segment wheat canopy from background soil and other objects.
This model produced pixel-level binary masks (1 = canopy, 0 = background)
which is used for downstream vegetation index (VI) calculation.

* **Training Dataset:** Canopy-labeled RGB-NIR images captured from Raspberry Pi NoIR v2 camera.
* **Architecture:** DeepLabV3+ (Chen et al., 2018) with a MobileNetV2 (Sandler et al., 2018) backbone.
* **High Efficiency:** Uses depth wise separable convolutions to maintain a low parameter count (<5M),
optimizing memory overhead for edge devices and limited VRAM systems.
* **Geometric Integrity:** Trained entirely on native rectangular aspect ratios ($640 \times 928$) to eliminate squashing
distortions and edge blur caused by traditional square resizing ($640 \times 640$).

---

## Test Performance

* **mean Intersection over Union (mIoU)**: 92.4%
* **F1-score**: 95.9%