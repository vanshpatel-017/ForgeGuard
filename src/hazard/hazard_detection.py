import cv2

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
# DANGER ZONE
# ============================================================

def create_danger_zone(box, padding=100):

    x1, y1, x2, y2 = box

    return [

        x1 - padding,
        y1 - padding,

        x2 + padding,
        y2 + padding
    ]

# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(TRAINED_MODEL))

OUTPUT_VIDEO = (
    OUTPUT_VIDEO_DIR /
    "04_hazard_intelligence_demo.mp4"
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

    boxes = results[0].boxes

    persons = []
    machinery = []

    for box in boxes:

        cls = int(box.cls[0])

        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        if CLASS_NAMES[cls] == "Person":
            persons.append(xyxy)

        elif CLASS_NAMES[cls] in ["machinery", "vehicle"]:
            machinery.append(xyxy)

    for machine in machinery:

        zone = create_danger_zone(machine)

        zx1, zy1, zx2, zy2 = zone

        cv2.rectangle(
            frame,
            (zx1, zy1),
            (zx2, zy2),
            (0, 0, 255),
            2
        )

        for person in persons:

            if calculate_iou(zone, person) > 0:

                px1, py1, px2, py2 = person

                cv2.rectangle(
                    frame,
                    (px1, py1),
                    (px2, py2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "HAZARD",
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    writer.write(frame)

cap.release()
writer.release()

print("Hazard intelligence complete.")