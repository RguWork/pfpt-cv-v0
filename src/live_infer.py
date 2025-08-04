import cv2, torch, collections, numpy as np, pathlib
from mediapipe_adapter import extract_kps
from utils import centre_scale
from model import BiLSTMClassifier

#configuration
WIN   = 64
STEP  = 16 #smaller value for smoother updates
LABEL = ["bg", "leg_abduction", "lunge", "squat"]

device = ("cuda" if torch.cuda.is_available()
          else "mps"  if torch.backends.mps.is_available()
          else "cpu")

#load model
MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "bilstm_live.pt"
model = torch.jit.load(MODEL_PATH, map_location=device).eval()

#sliding window queue for live camera
buffer = collections.deque(maxlen=WIN) #if exceeds 64 frames, every new appended frame pops the old one
frame_count = 0

cap = cv2.VideoCapture(0) # default webcam
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    kps = extract_kps(frame)
    buffer.append(kps if kps is not None else np.zeros((17, 2), dtype=np.float32))

    #every STEP frames, run inference
    if len(buffer) == WIN and frame_count % STEP == 0:
        clip = centre_scale(np.stack(buffer)) # (64,17,2)
        clip = torch.tensor(clip, dtype=torch.float32).flatten(1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(clip).argmax(1).item()
        cv2.putText(frame, LABEL[pred], (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("PF-PT demo  -  ESC to quit", frame)
    if cv2.waitKey(1) & 0xFF == 27:          # ESC
        break

cap.release()
cv2.destroyAllWindows()
