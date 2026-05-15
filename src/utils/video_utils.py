import cv2

# ============================================================
# GET VIDEO INFO
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(str(video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    cap.release()

    return width, height, fps

# ============================================================
# CREATE VIDEO WRITER
# ============================================================

def create_video_writer(output_path, width, height, fps):

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    return writer