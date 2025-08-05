import numpy as np

#mapping rehash to custom joint subset pfpt-15.
MAP_26_TO_PFPT15 = {
    0: 0,     # pelvis
    2: 1,     # mid_spine
    4: 2,     # head
    7: 3,     # L_shoulder
    8: 4,     # L_elbow
    9: 5,     # L_wrist
    12:6,     # R_shoulder
    13:7,     # R_elbow
    14:8,     # R_wrist
    16:9,     # L_hip
    17:10,    # L_knee
    18:11,    # L_ankle
    21:12,    # R_hip
    22:13,    # R_knee
    23:14,    # R_ankle
}


PELVIS = 0
MIDSPINE = 1
LSHO, RSHO = 3, 6

def to_pfpt15(x26):
    """(T,26,2) → (T,15,2) with NaNs for missing joints.""" 
    out = np.full((x26.shape[0], 15, 2), np.nan, np.float32)
    for src, dst in MAP_26_TO_PFPT15.items():
        out[:, dst] = x26[:, src]
    return out


def centre_scale(x15):
    """Pelvis to (0,0); divide by shoulder width each frame. Normalization."""
    pelvis = x15[:, PELVIS:PELVIS+1]
    centred = x15 - pelvis
    sh_width = np.linalg.norm(
        centred[:, LSHO] - centred[:, RSHO], axis=-1, keepdims=True) + 1e-6
    return centred / sh_width[..., None]