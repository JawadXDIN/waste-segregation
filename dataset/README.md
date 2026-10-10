Expected YOLO dataset:

dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
  dataset.yaml

Every image needs a matching .txt annotation file.

Example:
images/train/img001.jpg
labels/train/img001.txt

Label format:
class_id x_center y_center width height

All box coordinates are normalized 0..1.
