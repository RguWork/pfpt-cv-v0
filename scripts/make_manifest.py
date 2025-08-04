from pathlib import Path
import json, numpy as np
root = Path("../data/processed_data/windows")
paths = sorted(root.glob("*.npz"))
records = []
for p in paths:
    lbl  = int(np.load(p)["label"])
    qual = int(np.load(p)["quality"])
    records.append({"path": str(p), "label": lbl, "quality": qual})
Path(root.parent/"manifest.json").write_text(json.dumps(records, indent=2))