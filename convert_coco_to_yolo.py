"""
@author: Lochana Marasinghe
@date: 4/27/2026
@description: 
"""
from ultralytics.data.converter import convert_coco


convert_coco(labels_dir='coco_annotated_data/train', use_segments=True, save_dir='yolo_annotations/train')
convert_coco(labels_dir='coco_annotated_data/valid', use_segments=True, save_dir='yolo_annotations/valid')
convert_coco(labels_dir='coco_annotated_data/test', use_segments=True, save_dir='yolo_annotations/test')