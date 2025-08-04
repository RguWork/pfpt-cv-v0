from pathlib import Path
import json, numpy as np
root = Path("../data/processed_data/windows")
paths = sorted(root.glob("*.npz"))
records = []
LABEL_MAP = {0: 0, 4: 1, 5: 2, 6: 3}

for p in paths:
    lbl_raw = int(np.load(p)["label"])
    lbl = LABEL_MAP[lbl_raw]
    qual = int(np.load(p)["quality"])
    records.append({"path": str(p), "label": lbl, "quality": qual})
Path(root.parent/"manifest.json").write_text(json.dumps(records, indent=2))