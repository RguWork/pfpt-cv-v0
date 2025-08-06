import cv2
import numpy as np
import mediapipe as mp

#TEMP
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

#TEMPO


#the actual mediapipe pose estimator. initialise once – keep it global so the graph stays on the GPU / CPU
_mp_pose = mp.solutions.pose.Pose(static_image_mode=False,
                                  model_complexity=1,
                                  enable_segmentation=False)


def extract_kps(frame_bgr: np.ndarray):
    """
    Convert an OpenCV BGR frame → 15-joint pose array compatible with our
    BiLSTM (shape: (15,2), values in [0,1] image-relative coordinates).

    Returns (15,2) array in [0,1] coords or None if pose not detected.
    """
    res = _mp_pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None


    xyz33 = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark],
                     dtype=np.float32)  # (33,2)

    pelvis = (xyz33[23] + xyz33[24]) / 2
    mid_spine = (pelvis + xyz33[0]) / 2
    pfpt15 = np.stack([
        pelvis,
        mid_spine,
        xyz33[0],    # nose
        xyz33[11], xyz33[13], xyz33[15],  # L shoulder, elbow, wrist
        xyz33[12], xyz33[14], xyz33[16],  # R shoulder, elbow, wrist
        xyz33[23], xyz33[25], xyz33[27],  # L hip, knee, ankle
        xyz33[24], xyz33[26], xyz33[28],  # R hip, knee, ankle
    ])
    return pfpt15
