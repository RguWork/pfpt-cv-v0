#!/usr/bin/env python3
import json, numpy as np, torch
from pathlib import Path

from utils import centre_scale

# load your scripted model
model = torch.jit.load("../artifacts/bilstm_live_8_4.pt", map_location="cpu").eval()
labels = ["bg", "leg_abduction", "lunge", "squat"]

# load the manifest
manifest = json.load(open("../data/processed_data/manifest.json"))

def test_window(rec):
    xyz = np.load(rec["path"])["xyz"]         # (64,17,2)
    clip = centre_scale(xyz)                  # same preprocess as live
    inp = torch.tensor(clip, dtype=torch.float32)\
               .view(1, clip.shape[0], -1)    # (1,64,34)

    with torch.no_grad():
        probs = model(inp).softmax(1)[0].numpy()
    pred = probs.argmax()
    print(f"\nFile: {Path(rec['path']).name}")
    print(f"GT   : {rec['label']} → {labels[rec['label']]}")
    print(f"Pred : {pred} → {labels[pred]}")
    print("Probs:", {labels[i]: round(float(p),3) for i,p in enumerate(probs)})

# 1) exercise example
ex_rec = next(r for r in manifest if r["label"] == 1)  # pick leg_abduction
test_window(ex_rec)

# 2) background example
bg_rec = next(r for r in manifest if r["label"] == 0)  # pick the first bg
test_window(bg_rec)
