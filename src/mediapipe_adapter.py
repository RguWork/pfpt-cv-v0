

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

#convert mediapipe 33 body points to coco-17 subset
COCO17_FROM_MP = [
    23,                 # 0 pelvis (use left-hip as proxy; centre later)
    23, 25, 27,         # 1-3 L-hip / knee / ankle
    24, 26, 28,         # 4-6 R-hip / knee / ankle
    11,                 # 7 upper-spine/chest (use L-shoulder as proxy)
    11, 13,             # 8-9  L-shoulder / L-elbow
    12, 14,             # 10-11 R-shoulder / R-elbow
    16,                 # 12 R-wrist
    0,                  # 13 head-top  (nose)
    23,                 # 14 mid-spine (hip proxy)
    11,                 # 15 L-shoulder (dup)
    0                   # 16 head (dup)
]


def extract_kps(frame_bgr: np.ndarray):
    """
    Convert an OpenCV BGR frame → 17-joint pose array compatible with our
    BiLSTM (shape: (17,2), values in [0,1] image-relative coordinates).

    Returns (17,2) array in [0,1] coords or None if pose not detected.
    """
    res = _mp_pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None
    

    # # ─── draw full 33-landmark skeleton on the original frame ───
    # if res.pose_landmarks:
    #     mp_drawing.draw_landmarks(
    #         frame_bgr,                         # OpenCV BGR image
    #         res.pose_landmarks,
    #         mp.solutions.pose.POSE_CONNECTIONS,
    #         landmark_drawing_spec = mp_style.get_default_pose_landmarks_style()
    #     )
    # else:
    #     return None          # early exit → no pose

    xyz33 = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark],
                     dtype=np.float32) #(33,2)
    xyz17 = xyz33[COCO17_FROM_MP] #(17,2)
    return xyz17
