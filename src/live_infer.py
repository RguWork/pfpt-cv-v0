import time
import cv2
import torch
import collections
import numpy as np
import pathlib

from mediapipe_adapter import extract_kps
from utils import centre_scale
from model import BiLSTMClassifier

# ─── configuration ───────────────────────────────────────────────────────
WIN          = 64
STEP         = 8      # hop size between inferences
LABEL        = ["bg", "leg_abduction", "lunge", "squat"]
SMOOTH       = 3
VALID_JOINTS = 12     # require at least this many visible joints

def count_valid_joints(kps):
    return int(np.count_nonzero(np.any(kps != 0, axis=1)))

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ─── load model ───────────────────────────────────────────────────────────
MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "bilstm_live_8_4.pt"
model = torch.jit.load(MODEL_PATH, map_location=device).eval()

# ─── state ───────────────────────────────────────────────────────────────
last_pred   = 0
vote_buf    = collections.deque(maxlen=SMOOTH)
buffer      = collections.deque(maxlen=WIN)
start_time  = time.time()
got_ready   = False
frame_count = 0
prev_kps = None

# ─── open webcam ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    elapsed = time.time() - start_time
    if not got_ready:
        # show countdown
        secs_left = max(0, int(3 - elapsed) + 1)
        cv2.putText(frame,
                    f"Get ready: {secs_left}s",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    2)
        cv2.imshow("PF-PT demo  -  ESC to quit", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if elapsed >= 3:
            got_ready = True
        continue

    # ─── adapter & buffer ────────────────────────────────────────────────
    kps = extract_kps(frame)

    if prev_kps is None:
        if kps is None or count_valid_joints(kps) < VALID_JOINTS:
            # still no seed → skip buffering entirely
            continue
        else:
            # first good frame: clamp & store it
            kps = np.clip(kps, 0.0, 1.0)
            prev_kps = kps.copy()
            use_kps = prev_kps

    # 2) For subsequent frames, carry‐forward or fill‐in
    else:
        if kps is None or count_valid_joints(kps) < VALID_JOINTS:
            # reuse the last known good skeleton
            use_kps = prev_kps.copy()
        else:
            # clamp coordinates
            kps = np.clip(kps, 0.0, 1.0)
            # fill any missing joints from prev_kps
            mask = np.all(kps == 0, axis=1)
            kps[mask] = prev_kps[mask]
            prev_kps = kps.copy()
            use_kps = kps


    buffer.append(use_kps)

    # ─── inference every STEP frames ─────────────────────────────────────
    if len(buffer) == WIN and frame_count % STEP == 0:
        clip = centre_scale(np.stack(buffer))   # (64,17,2)

        arr = torch.tensor(clip, dtype=torch.float32)         # (64,17,2)
        feat_live = arr.view(1, WIN, -1)[0,0].cpu().numpy()   # first time‐step → length-34

        # load a ref window’s first frame for comparison
        ref = np.load("../data/processed_data/windows/PM_005_bg_c17_00000.npz")["xyz"]
        ref_feat = centre_scale(ref)[0].reshape(-1)

        names = JOINT_NAMES_34 = [
            "pelvis_x",    "pelvis_y",
            "l_hip_x",     "l_hip_y",
            "l_knee_x",    "l_knee_y",
            "l_ankle_x",   "l_ankle_y",
            "r_hip_x",     "r_hip_y",
            "r_knee_x",    "r_knee_y",
            "r_ankle_x",   "r_ankle_y",
            "chest_x",     "chest_y",
            "l_shoulder_x","l_shoulder_y",
            "l_elbow_x",   "l_elbow_y",
            "r_shoulder_x","r_shoulder_y",
            "r_elbow_x",   "r_elbow_y",
            "r_wrist_x",   "r_wrist_y",
            "head_top_x",  "head_top_y",
            "mid_spine_x", "mid_spine_y",
            "l_shoulder2_x","l_shoulder2_y",
            "head2_x",     "head2_y",
        ]
        for i,(n,v_live,v_ref) in enumerate(zip(names, feat_live, ref_feat)):
            print(f"{i:2d} {n:10s}  live={v_live:.3f}   ref={v_ref:.3f}")

        if np.isnan(clip).any():
            print("NaNs in clip - skipping inference")
        else:
            # reshape → (1, 64, 34)
            arr = torch.tensor(clip, dtype=torch.float32).to(device)
            inp = arr.view(1, WIN, -1)

            with torch.no_grad():
                pred = model(inp).argmax(1).item()

            # update via majority vote
            print(LABEL[pred])
            vote_buf.append(pred)
            majority = max(set(vote_buf), key=vote_buf.count)
            last_pred = majority

    # ─── draw the 17-joint overlay ───────────────────────────────────────
    if kps is not None:
        H, W = frame.shape[:2]
        pts  = (kps * [W, H]).astype(int)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        LIMBS = [
            (0,1),(1,2),(2,3),
            (0,4),(4,5),(5,6),
            (7,8),(8,9),
            (7,10),(10,11),(11,12),
        ]
        for i,j in LIMBS:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), (0,128,0), 2)

    # ─── draw the persistent label ────────────────────────────────────────
    cv2.putText(frame,
                LABEL[last_pred],
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2)

    cv2.imshow("PF-PT demo  -  ESC to quit", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
