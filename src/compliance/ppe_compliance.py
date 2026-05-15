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
# IOU CALCULATION
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
    "03_ppe_compliance_demo.mp4"
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
    helmets = []
    vests = []

    for box in boxes:

        cls = int(box.cls[0])

        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        if CLASS_NAMES[cls] == "Person":
            persons.append(xyxy)

        elif CLASS_NAMES[cls] == "Hardhat":
            helmets.append(xyxy)

        elif CLASS_NAMES[cls] == "Safety Vest":
            vests.append(xyxy)

    for person in persons:

        helmet_ok = False
        vest_ok = False

        for helmet in helmets:

            if calculate_iou(person, helmet) > 0:
                helmet_ok = True

        for vest in vests:

            if calculate_iou(person, vest) > 0:
                vest_ok = True

        x1, y1, x2, y2 = person

        if helmet_ok and vest_ok:

            color = (0, 255, 0)
            label = "SAFE"

        else:

            color = (0, 0, 255)
            label = "VIOLATION"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    writer.write(frame)

cap.release()
writer.release()

print("PPE compliance complete.")