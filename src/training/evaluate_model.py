from ultralytics import YOLO

from src.utils.constants import (
    TRAINED_MODEL,
    YAML_PATH
)

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(TRAINED_MODEL))

# ============================================================
# VALIDATE
# ============================================================

metrics = model.val(
    data=str(YAML_PATH)
)

print("\n========== FINAL METRICS ==========")

print(f"mAP50      : {metrics.box.map50:.4f}")
print(f"mAP50-95   : {metrics.box.map:.4f}")
print(f"Precision  : {metrics.box.mp:.4f}")
print(f"Recall     : {metrics.box.mr:.4f}")