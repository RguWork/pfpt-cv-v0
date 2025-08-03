import numpy as np, pandas as pd, json
from pathlib import Path
from utils import to_coco17, centre_scale

RAW  = Path("../data/raw_data")
OUT  = Path("../data/processed_data/windows");  OUT.mkdir(parents=True, exist_ok=True) #OUR.mkdir to make a directory


#define the temporal frame the LSTM will see
WIN  = 64     #frames per window
STEP = 32     #50 % overlap

def window(arr, label, qual, vid, cam, base_idx):
    """
    save an npz file that is normalized and turned to coco17
    with batched frames, every 64 frames with 32 frame overlap
    """
    #normalize and turn to coco17 each window of exercise
    #note: we save len 64 frames every 32 frame steps for a 32 frame overlap
    for start in range(0, len(arr) - WIN + 1, STEP):
        start_global = base_idx + start 
        #len(arr) - WIN  + 1 to avoid out of bounds
        clip = arr[start:start+WIN]              # (64,26,2), frame x joints x dimensions (x,y)
        clip17 = centre_scale(to_coco17(clip))   # (64,17,2)
        idx = f"{vid}_{cam}_{start_global:05d}"
        np.savez_compressed(OUT/f"{idx}.npz",
                            xyz=clip17.astype(np.float32),
                            label=label,
                            quality=qual)
        print("saving", OUT/f"{idx}.npz")


def process_video(ex_folder, fname):
    """
    takes an npy file, and partitions it into frames when the exercise is actually
    being done, and background frames, then call window on both
    """
    vid, cam, _ = fname.stem.split('-')     # PM_000, c17 etc.
    arr = np.load(fname)                        # (T,26,2)
    seg = SEG[SEG.video_id == vid]

    #mask to keep track of booleans of inside-rep framse
    mask = np.zeros(len(arr), bool)

    for row in seg.itertuples():
        if row.exercise_id not in PFPT_IDS:     # skip non-PFPT
            #DEBUGGING SCRIPTS
            SKIPPED[row.exercise_id] = SKIPPED.get(row.exercise_id, 0) + 1
            continue
        PROCESSED[row.exercise_id] = PROCESSED.get(row.exercise_id, 0) + 1

        mask[row.first_frame:row.last_frame+1] = True
        rep = arr[row.first_frame:row.last_frame+1]
        window(rep, row.exercise_id, row.correctness, vid, cam, row.first_frame)

    #background windows
    bg_frames = arr[~mask] #make an array of all non-exercise frames
    window(bg_frames, 0, -1, f"{vid}_bg", cam, 0)

if __name__ == "__main__":
    global SEG, PFPT_IDS
    SEG = pd.read_csv(RAW/"Segmentation.csv", sep=';')
    SEG["exercise_id"] = SEG.exercise_id.astype(int)
    PFPT_IDS = [4,5,6]          #our 3 PFPT exercises
    SKIPPED, PROCESSED = {}, {}


    rep_counts = SEG[SEG.exercise_id.isin(PFPT_IDS)].exercise_id.value_counts()



    for ex in ("Ex4","Ex5","Ex6"):
        for f in (RAW/"2d_joints"/ex).glob("*-30fps.npy"):
            process_video(ex, f)

    meta = dict(window=WIN, step=STEP, mapping="26→17 coco", fps=30)
    (OUT.parent/"meta.json").write_text(json.dumps(meta, indent=2))
    print("✓ windows saved to", OUT)
    print("processed reps :", PROCESSED)
    print("skipped reps   :", SKIPPED)
