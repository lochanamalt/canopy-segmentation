"""
@author: Lochana Marasinghe
@date: 2/27/2026
@description: 
"""
from tensorflow.python.tools.saved_model_cli import command_required_flags

from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = "deeplab_canopy_seg_preprocessed.onnx"
output_model = "deeplab_canopy_seg_int8.onnx"

quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8  # INT8 weights
)

print("Quantized model saved:", output_model)