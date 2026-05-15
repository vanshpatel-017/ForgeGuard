import cv2

from ultralytics import YOLO

from src.utils.constants import (
    TRAINED_MODEL,
    INPUT_VIDEO,
    OUTPUT_VIDEO_DIR
)

from src.utils.video_utils import (
    get_video_info,
    create_video_writer
)

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(TRAINED_MODEL))

# ============================================================
# OUTPUT VIDEO
# ============================================================

OUTPUT_VIDEO = (
    OUTPUT_VIDEO_DIR /
    "02_tracking_demo.mp4"
)

# ============================================================
# VIDEO INFO
# ============================================================

width, height, fps = get_video_info(INPUT_VIDEO)

cap = cv2.VideoCapture(str(INPUT_VIDEO))

writer = create_video_writer(
    OUTPUT_VIDEO,
    width,
    height,
    fps
)

# ============================================================
# TRACKING LOOP
# ============================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        source=frame,
        persist=True,
        conf=0.35,
        verbose=False
    )

    plotted = results[0].plot()

    writer.write(plotted)

cap.release()
writer.release()

print("Tracking complete.")