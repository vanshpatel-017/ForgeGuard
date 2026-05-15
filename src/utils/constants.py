from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# MODEL PATHS
# ============================================================

TRAINED_MODEL = (
    PROJECT_ROOT /
    "models" /
    "trained" /
    "ppe_yolo11m_best.pt"
)

PRETRAINED_MODEL = (
    PROJECT_ROOT /
    "models" /
    "pretrained" /
    "yolo11m.pt"
)

# ============================================================
# DATASET
# ============================================================

DATASET_PATH = (
    PROJECT_ROOT /
    "datasets" /
    "processed" /
    "dataset_final"
)

YAML_PATH = DATASET_PATH / "data.yaml"

# ============================================================
# VIDEO PATHS
# ============================================================

INPUT_VIDEO = (
    PROJECT_ROOT /
    "videos" /
    "input" /
    "input.mp4"
)

OUTPUT_VIDEO_DIR = (
    PROJECT_ROOT /
    "videos" /
    "output"
)

# ============================================================
# OUTPUTS
# ============================================================

PLOTS_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "plots"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "predictions"
)

# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle"
}