from ultralytics import YOLO
import torch

from src.utils.constants import (
    PRETRAINED_MODEL,
    YAML_PATH
)

print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(PRETRAINED_MODEL))

# ============================================================
# TRAIN MODEL
# ============================================================

model.train(

    data=str(YAML_PATH),

    epochs=100,

    imgsz=960,

    batch=4,

    device=0,

    workers=4,

    cache=True,

    optimizer="AdamW",

    lr0=0.001,

    lrf=0.01,

    weight_decay=0.0005,

    momentum=0.937,

    cos_lr=True,

    dropout=0.05,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    degrees=10.0,

    translate=0.1,

    scale=0.5,

    shear=2.0,

    perspective=0.0005,

    fliplr=0.5,

    mosaic=1.0,

    mixup=0.15,

    copy_paste=0.1,

    close_mosaic=10,

    overlap_mask=True,

    amp=True,

    pretrained=True,

    patience=20,

    project="ppe_safety_system",

    name="yolo11m_960"
)