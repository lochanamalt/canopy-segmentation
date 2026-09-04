"""
@author: Lochana Marasinghe
@date: 9/4/2026
@description: 
"""

import torch
import segmentation_models_pytorch as smp

def export_deeplab_to_onnx(weights_path: str, output_onnx_path: str):
    DEVICE = "cpu"

    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(DEVICE)

    # Load trained weights
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    model.eval()

    dummy_input = torch.randn(1, 3, 928, 640, dtype=torch.float32)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,  # Stable opset for modern vision layers
        do_constant_folding=True,  # Optimizes out fixed weights
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # Optional: allows variable batch sizes
            "output": {0: "batch_size"}
        }
    )
    print(f"Successfully exported model to {output_onnx_path}")


if __name__ == "__main__":
    export_deeplab_to_onnx("models/best_canopy_seg_model.pth", "models/deeplab_canopy_seg.onnx")