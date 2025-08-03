import numpy as np

#map subset of coco-17: 17 joints from the 26 given by REHAB
MAP_26_TO_17 = {
    #OptiTrack-26 index : COCO-17 index
    0: 0,   # pelvis / hips
    16: 1, 17: 2, 18: 3,          # L hip, knee, ankle
    23: 4, 24: 5, 25: 6,          # R hip, knee, ankle
    4: 7,  5: 8,  6: 9,           # spine-shoulder-L wrist
    11:10, 12:11,                 # R shoulder, R elbow
    13:12, 14:13,                 # R wrist, head top
    1:14,                         # mid-spine
    7:15,  9:16                   # L shoulder, head (alt)
}

PELVIS = 0 
LSHO, RSHO = 7, 10

def to_coco17(x26):
    """(T,26,2) → (T,17,2) with NaNs for missing joints.""" 
    out = np.full((x26.shape[0], 17, 2), np.nan, np.float32)
    for src, dst in MAP_26_TO_17.items():
        out[:, dst] = x26[:, src]
    return out


def centre_scale(x17):
    """Pelvis to (0,0); divide by shoulder width each frame. Normalization."""
    #subtract pelvis from eveyrthing to make it 0,0
    pelvis = x17[:, PELVIS:PELVIS+1]
    centred = x17 - pelvis
    sh_width = np.linalg.norm(
        centred[:, LSHO] - centred[:, RSHO], axis=-1, keepdims=True) + 1e-6
    return centred / sh_width[..., None]