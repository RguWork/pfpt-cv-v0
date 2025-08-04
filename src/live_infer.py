import cv2, torch, collections, numpy as np, pathlib
from mediapipe_adapter import extract_kps
from utils import centre_scale
from model import BiLSTMClassifier

#configuration
WIN   = 64
STEP  = 8 #smaller value for smoother updates
LABEL = ["bg", "leg_abduction", "lunge", "squat"]
SMOOTH = 3

device = ("cuda" if torch.cuda.is_available()
          else "mps"  if torch.backends.mps.is_available()
          else "cpu")

#load model
MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "bilstm_live_8_4.pt"
model = torch.jit.load(MODEL_PATH, map_location=device).eval()

#label buffer queue
last_pred   = 0
vote_buf    = collections.deque(maxlen=SMOOTH)


#sliding window queue for live camera
buffer = collections.deque(maxlen=WIN) #if exceeds 64 frames, every new appended frame pops the old one
frame_count = 0

cap = cv2.VideoCapture(0) # default webcam
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_count += 1

    kps = extract_kps(frame)
    buffer.append(kps if kps is not None else np.zeros((17, 2), dtype=np.float32))

    #every STEP frames, run inference
    if len(buffer) == WIN and frame_count % STEP == 0:
        clip = centre_scale(np.stack(buffer)) # (64,17,2)

        #DEBUG 1
        if np.isnan(clip).any():
            print("NaNs in clip - pose missing too often")

        else:
            clip = torch.tensor(clip, dtype=torch.float32).flatten(1).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(clip).argmax(1).item()


            #label changing logic
            vote_buf.append(pred)
            majority = max(set(vote_buf), key=vote_buf.count)

            if majority != last_pred:
                last_pred = majority

            print(LABEL[pred])

    cv2.putText(frame, LABEL[last_pred], (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("PF-PT demo  -  ESC to quit", frame)
    if cv2.waitKey(1) & 0xFF == 27:          # ESC
        break

cap.release()
cv2.destroyAllWindows()
