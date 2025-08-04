# %%
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
from collections import defaultdict


from dataset import PoseWindowDataset
from model import BiLSTMClassifier


# %%
ROOT_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT_DIR / "artifacts"
SAVE_DIR.mkdir(exist_ok=True)

# %%
MANIFEST_PATH = Path("../data/processed_data/manifest.json")
mf = pd.read_json(MANIFEST_PATH)

train_pids = {1, 2, 3, 4, 5, 6, 7} #76.6%
# val_pids   = {8} #15.5%
# test_pids  = {9} #7.9%

#SWAPPED FOR NOW
val_pids   = {9}
test_pids  = {8}

pid2bucket = ({pid: "train" for pid in train_pids} |
              {pid: "val"   for pid in val_pids}   |
              {pid: "test"  for pid in test_pids})

train_idx, val_idx, test_idx = [], [], []
for idx, row in mf.iterrows():
    bucket = pid2bucket[row.person_id]
    (train_idx if bucket == "train" else
     val_idx   if bucket == "val"   else
     test_idx).append(idx)

print(f"#windows  train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")

dataset  = PoseWindowDataset(MANIFEST_PATH)
train_ds = torch.utils.data.Subset(dataset, train_idx)
val_ds   = torch.utils.data.Subset(dataset, val_idx)
test_ds  = torch.utils.data.Subset(dataset, test_idx)
# %%

#class weights for exercise loss (labels 0,1,2,3: background, ex4, ex5, ex6)
labels = torch.tensor([dataset[idx][1] for idx in train_ds.indices]) #labels for training
counts = torch.bincount(labels, minlength=4).float() #counts for each class
weights = 1 / counts #determine weights to scale the loss, compensating for larger background class 
weights[counts == 0] = 0

# --- sanity log ------------------------------------------------------------
print("class counts :", counts.tolist())
print("class weights:", weights.tolist())
# --------------------------------------------------------------------------

exercise_criterion = torch.nn.CrossEntropyLoss(weight=weights.to(torch.float32))

#leverage a weighted sampler to oversample the "rarer" classes (1,2,3)
sample_weights = weights[labels]

sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True) 

#TODO: workers at 0 for now, change to 4 when training in a main() function
train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,    num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,    num_workers=0)

# %%
#helpers
def run_epoch(model, loader, criterion, optimizer=None):
    """
    Train when optimizer is given, else evaluate.
    Returns avg_loss, overall_acc, per-class F1 dict.
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    all_preds, all_lbls = [], []
    total_loss, n = 0.0, 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)
        out = model(x) # (B,4)
        loss = criterion(out, y)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
        all_preds.append(out.argmax(1).cpu())
        all_lbls.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_lbls  = torch.cat(all_lbls)
    acc = (all_preds == all_lbls).float().mean().item()
    f1 = f1_score(all_lbls, all_preds, average=None, labels=[0,1,2,3])
    return total_loss / n, acc, {i: f for i, f in enumerate(f1)}

# %%
#training loop

device = ("cuda" if torch.cuda.is_available()
          else "mps"  if torch.backends.mps.is_available()
          else "cpu") #training locally on mac
print(device)
model = BiLSTMClassifier().to(device)
exercise_criterion = exercise_criterion.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, verbose=True)

EPOCHS      = 50
PATIENCE    = 7 # early-stop if val loss ↑ for 7 epochs
best_val    = float("inf")
counter_bad = 0
log_history = defaultdict(list)


for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, _      = run_epoch(model, train_loader, exercise_criterion, optimizer)
    val_loss, val_acc, f1s  = run_epoch(model, val_loader,   exercise_criterion)

    scheduler.step(val_loss)

    # --- logging ---------------------------------------------------------
    print(f"[{epoch:02d}] "
          f"train L {tr_loss:.4f}  acc {tr_acc:.3f} | "
          f"val L {val_loss:.4f}  acc {val_acc:.3f} "
          f"F1 {f1s}")

    for k, v in zip(
        ["tr_loss","tr_acc","val_loss","val_acc"], 
        [tr_loss, tr_acc, val_loss, val_acc]
    ):
        log_history[k].append(v)
    log_history["f1"].append(f1s)

    # --- early-stopping --------------------------------------------------
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        counter_bad = 0
        torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
    else:
        counter_bad += 1
        if counter_bad >= PATIENCE:
            print("Early stopping triggered.")
            break
# %%

#testing evaluation

model.load_state_dict(torch.load(SAVE_DIR / "best_model.pt"))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y, _ in test_loader:
        x = x.to(device, non_blocking=(device == "cuda"))
        y = y.to(device, non_blocking=(device == "cuda"))

        logits = model(x)
        y_true.append(y.cpu())
        y_pred.append(logits.argmax(1).cpu())

y_true = torch.cat(y_true)
y_pred = torch.cat(y_pred)

test_loss, test_acc, test_f1 = run_epoch(
    model, test_loader, exercise_criterion, optimizer=None
)

print(f"\nTest metrics  ---------------------------")
print(f"loss {test_loss:.4f}  acc {test_acc:.3f}")
print("F1 per class:", test_f1)
print("Detailed report:\n",
      classification_report(
          y_true, y_pred,
          target_names=["bg", "abduction", "lunge", "squat"]))
# %%
