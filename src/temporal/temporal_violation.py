import cv2
import time

from ultralytics import YOLO

from src.utils.constants import (
    TRAINED_MODEL,
    INPUT_VIDEO,
    OUTPUT_VIDEO_DIR,
    CLASS_NAMES
)

from src.utils.video_utils import (
    get_video_info,
    create_video_writer
)

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(TRAINED_MODEL))

OUTPUT_VIDEO = (
    OUTPUT_VIDEO_DIR /
    "05_temporal_violations_demo.mp4"
)

width, height, fps = get_video_info(INPUT_VIDEO)

cap = cv2.VideoCapture(str(INPUT_VIDEO))

writer = create_video_writer(
    OUTPUT_VIDEO,
    width,
    height,
    fps
)

# ============================================================
# TEMPORAL MEMORY
# ============================================================

violation_memory = {}

VIOLATION_THRESHOLD = 5

# ============================================================
# MAIN LOOP
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

    boxes = results[0].boxes

    for box in boxes:

        cls = int(box.cls[0])

        if box.id is None:
            continue

        track_id = int(box.id[0])

        label = CLASS_NAMES[cls]

        if label in [

            "NO-Hardhat",
            "NO-Mask",
            "NO-Safety Vest"
        ]:

            if track_id not in violation_memory:

                violation_memory[track_id] = time.time()

            duration = (
                time.time() -
                violation_memory[track_id]
            )

            if duration > VIOLATION_THRESHOLD:

                x1, y1, x2, y2 = (

                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                cv2.rectangle(
                    plotted,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    plotted,
                    "TEMPORAL VIOLATION",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

    writer.write(plotted)

cap.release()
writer.release()

print("Temporal monitoring complete.")