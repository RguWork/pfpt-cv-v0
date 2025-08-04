from pathlib import Path
import json, numpy as np, re

root = Path("../data/processed_data/windows")
paths = sorted(root.glob("*.npz"))
records = []
LABEL_MAP = {0: 0, 4: 1, 5: 2, 6: 3}


def vid_from_path(p: str) -> str:
    """
    './windows/PM_117a_bg_c17_00000.npz' -> 'PM_117'
    './windows/PM_116_c18_00495.npz' -> 'PM_116'
    """
    stem = Path(p).stem #PM_117_bg_c17_00000
    m = re.match(r'(PM_\d+[a-z]?)', stem)
    return m.group(1) if m else '_'.join(stem.split('_')[:2])

for p in paths:
    vid = vid_from_path(p)
    lbl_raw = int(np.load(p)["label"])
    lbl = LABEL_MAP[lbl_raw]
    qual = int(np.load(p)["quality"])
    pid = int(np.load(p)["person_id"])
    records.append({"path": str(p), "label": lbl, "quality": qual, "video_id": vid, "person_id": pid})
Path(root.parent/"manifest.json").write_text(json.dumps(records, indent=2))