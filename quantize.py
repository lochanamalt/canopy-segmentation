"""
@author: Lochana Marasinghe
@date: 2/27/2026
@description:
 Do following before running this script:

 .pt to onnx conversion command:
 yolo export model=models/best_canopy_seg_model format=onnx  device=CPU simplify=True

 rename the resulted onnx file to deeplab_canopy_seg.onnx

 Then preprocess before quantization
 python -m onnxruntime.quantization.preprocess --input models/deeplab_canopy_seg.onnx --output models/deeplab_canopy_seg_preprocessed.onnx

"""

from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = "models/deeplab_canopy_seg_preprocessed.onnx"
output_model = "models/deeplab_canopy_seg_int8.onnx"

quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8
)

print("Quantized model saved:", output_model)