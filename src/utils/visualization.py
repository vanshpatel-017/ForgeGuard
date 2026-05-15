import cv2

# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(frame, text, position, color=(0, 255, 0)):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

# ============================================================
# DRAW BOX
# ============================================================

def draw_box(frame, box, color=(0, 255, 0)):

    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )