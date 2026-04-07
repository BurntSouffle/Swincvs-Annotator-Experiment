"""
Criterion-Specific Label Strategy (Mixed Model)
=================================================
Trains a single model where each criterion uses its optimal label source:
  C1 = Annotator 3 labels
  C2 = Majority Vote labels
  C3 = Ann2+3 AND labels

Evaluates against all label sets and produces a complete comparison table
with all 7 ViT models + SwinCVS baselines.
"""

import os
import sys
import ast
import time
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("C:/Users/sufia/Documents/Uni/Masters/DISSERTATION")
ENDOSCAPES_DIR = BASE_DIR / "endoscapes"
METADATA_CSV = ENDOSCAPES_DIR / "all_metadata.csv"
OUTPUT_DIR = BASE_DIR / "annotator_experiment" / "outputs_exp3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREV_DIR_1 = BASE_DIR / "annotator_experiment" / "outputs"       # exp1: ann1,ann2,ann3,mv
PREV_DIR_2 = BASE_DIR / "annotator_experiment" / "outputs_exp2"  # exp2: and23, or23

# ── Hyperparameters ──────────────────────────────────────────────────────────
INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 4
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERIA = ["C1", "C2", "C3"]

# All models for comparison
ALL_MODELS = ["ann1", "ann2", "ann3", "mv", "and23", "or23", "mixed"]
MODEL_NAMES = {
    "ann1": "Annotator 1", "ann2": "Annotator 2", "ann3": "Annotator 3",
    "mv": "Majority Vote", "and23": "Ann2+3 AND", "or23": "Ann2+3 OR",
    "mixed": "Mixed Strategy",
}

# Evaluation label sets
EVAL_LABEL_SETS = ["ann1", "ann2", "ann3", "mv", "and23", "or23", "mixed"]
EVAL_NAMES = {
    "ann1": "Annotator 1", "ann2": "Annotator 2", "ann3": "Annotator 3",
    "mv": "Majority Vote", "and23": "Ann2+3 AND", "or23": "Ann2+3 OR",
    "mixed": "Mixed",
}

# SwinCVS reference (per-criterion on MV labels)
SWINCVS_FROZEN = {"C1": 65.02, "C2": 61.38, "C3": 75.95, "mean": 67.45}
SWINCVS_E2E = {"C1": 64.23, "C2": 62.50, "C3": 67.03, "mean": 64.59}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_annotator_votes(vote_str):
    try:
        parsed = ast.literal_eval(vote_str)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
            return [int(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    return None


def get_split(vid):
    if vid <= 120: return "train"
    elif vid <= 161: return "val"
    else: return "test"


def identify_black_frames(df):
    black_set = set()
    print("Scanning for black frames...")
    records = list(zip(df["vid"].astype(int), df["frame"].astype(int)))
    splits = [get_split(v) for v, _ in records]
    for (vid, frame), split in tqdm(zip(records, splits), total=len(records),
                                     desc="Black frame scan"):
        img_path = ENDOSCAPES_DIR / split / f"{vid}_{frame}.jpg"
        if not img_path.exists():
            black_set.add((vid, frame))
            continue
        try:
            if img_path.stat().st_size < 10000:
                img = Image.open(img_path)
                if np.array(img).mean() == 0:
                    black_set.add((vid, frame))
        except Exception:
            black_set.add((vid, frame))
    print(f"  Found {len(black_set)} black/corrupted frames")
    return black_set


def load_and_prepare_data():
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)

    df = pd.read_csv(METADATA_CSV)
    df = df[df["is_ds_keyframe"] == True].copy()
    print(f"Total keyframes: {len(df)}")

    # Parse annotator votes
    for ann_idx in [1, 2, 3]:
        col = f"cvs_annotator_{ann_idx}"
        parsed = df[col].apply(parse_annotator_votes)
        df[f"C1_ann{ann_idx}"] = parsed.apply(lambda x: x[0] if x else np.nan)
        df[f"C2_ann{ann_idx}"] = parsed.apply(lambda x: x[1] if x else np.nan)
        df[f"C3_ann{ann_idx}"] = parsed.apply(lambda x: x[2] if x else np.nan)

    # Majority vote
    for c in CRITERIA:
        df[f"{c}_mv"] = (df[c] >= 0.5).astype(float)

    # AND strategy
    for c in CRITERIA:
        df[f"{c}_and23"] = ((df[f"{c}_ann2"] == 1) & (df[f"{c}_ann3"] == 1)).astype(float)

    # OR strategy
    for c in CRITERIA:
        df[f"{c}_or23"] = ((df[f"{c}_ann2"] == 1) | (df[f"{c}_ann3"] == 1)).astype(float)

    # MIXED strategy: C1=Ann3, C2=MV, C3=AND23
    df["C1_mixed"] = df["C1_ann3"]
    df["C2_mixed"] = df["C2_mv"]
    df["C3_mixed"] = df["C3_and23"]

    # Drop missing
    ann_cols = [f"{c}_ann{a}" for c in CRITERIA for a in [1, 2, 3]]
    df = df.dropna(subset=ann_cols)

    df["split"] = df["vid"].apply(get_split)

    # Exclude black frames
    black_frames = identify_black_frames(df)
    before = len(df)
    df = df[~df.apply(lambda r: (int(r["vid"]), int(r["frame"])) in black_frames, axis=1)]
    print(f"After excluding black frames: {len(df)} ({before - len(df)} excluded)")

    df["img_path"] = df.apply(
        lambda r: str(ENDOSCAPES_DIR / r["split"] / f"{int(r['vid'])}_{int(r['frame'])}.jpg"),
        axis=1
    )

    print(f"\nFinal dataset: {len(df)} frames")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {(df['split'] == split).sum()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT
# ═══════════════════════════════════════════════════════════════════════════════

def run_preflight(df):
    print("\n" + "=" * 70)
    print("PRE-FLIGHT: MIXED STRATEGY LABEL STATISTICS")
    print("=" * 70)

    print(f"\n{'Strategy':<18} {'C1+ rate':>10} {'C2+ rate':>10} {'C3+ rate':>10}")
    print("-" * 50)
    for ls in ["mv", "mixed"]:
        name = {"mv": "Majority Vote", "mixed": "Mixed Strategy"}[ls]
        rates = [df[f"{c}_{ls}"].mean() * 100 for c in CRITERIA]
        print(f"{name:<18} {rates[0]:>9.1f}% {rates[1]:>9.1f}% {rates[2]:>9.1f}%")

    print(f"\n  Mixed = C1 from Ann3 ({df['C1_ann3'].mean()*100:.1f}%), "
          f"C2 from MV ({df['C2_mv'].mean()*100:.1f}%), "
          f"C3 from AND23 ({df['C3_and23'].mean()*100:.1f}%)")

    # Spot check: show 5 frames where mixed differs from MV
    print("\n  Spot check (frames where mixed != MV):")
    diffs = df[
        (df["C1_mixed"] != df["C1_mv"]) |
        (df["C2_mixed"] != df["C2_mv"]) |
        (df["C3_mixed"] != df["C3_mv"])
    ].head(5)
    for _, row in diffs.iterrows():
        print(f"    vid={int(row['vid'])} frame={int(row['frame'])}: "
              f"MV=[{row['C1_mv']:.0f},{row['C2_mv']:.0f},{row['C3_mv']:.0f}] "
              f"Mixed=[{row['C1_mixed']:.0f},{row['C2_mixed']:.0f},{row['C3_mixed']:.0f}]")

    # Per-split breakdown
    print("\n  Per-split positive rates (Mixed):")
    for split in ["train", "val", "test"]:
        sdf = df[df["split"] == split]
        rates = [sdf[f"{c}_mixed"].mean() * 100 for c in CRITERIA]
        print(f"    {split}: C1={rates[0]:.1f}%  C2={rates[1]:.1f}%  C3={rates[2]:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET & MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class CVSDataset(Dataset):
    def __init__(self, df, label_set, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_cols = [f"{c}_{label_set}" for c in CRITERIA]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor([row[col] for col in self.label_cols], dtype=torch.float32)
        return img, labels


def get_transforms(is_train):
    if is_train:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def create_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    for param in model.parameters():
        param.requires_grad = False
    num_blocks = len(model.blocks)
    for i in range(num_blocks - 2, num_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True
    if hasattr(model, "norm"):
        for param in model.norm.parameters():
            param.requires_grad = True
    head = nn.Linear(model.num_features, 3)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    return nn.Sequential(model, head)


def compute_pos_weights(df, label_set):
    weights = []
    for c in CRITERIA:
        col = f"{c}_{label_set}"
        pos = df[col].sum()
        neg = len(df) - pos
        weights.append(neg / max(pos, 1))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model, dataloader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            logits = model(images)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_labels.append(labels.numpy())
            all_preds.append(preds)
    return np.concatenate(all_labels), np.concatenate(all_preds)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_mixed_model(df):
    print(f"\n{'='*70}")
    print("TRAINING MIXED STRATEGY MODEL")
    print(f"{'='*70}")
    print("  C1 supervised by: Annotator 3")
    print("  C2 supervised by: Majority Vote")
    print("  C3 supervised by: Ann2+3 AND")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_dataset = CVSDataset(train_df, "mixed", get_transforms(is_train=True))
    val_dataset = CVSDataset(val_df, "mixed", get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = create_model().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    pos_weight = compute_pos_weights(train_df, "mixed").to(DEVICE)
    print(f"  Pos weights: C1={pos_weight[0]:.2f}, C2={pos_weight[1]:.2f}, C3={pos_weight[2]:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_map = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_map": [],
               "val_c1_ap": [], "val_c2_ap": [], "val_c3_ap": [], "lr": []}

    save_dir = OUTPUT_DIR / "model_mixed"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_losses = []
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        val_labels, val_preds = evaluate_model(model, val_loader)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)

        val_aps = []
        for i in range(3):
            y_true, y_score = val_labels[:, i], val_preds[:, i]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                val_aps.append(float("nan"))
            else:
                val_aps.append(average_precision_score(y_true, y_score))
        val_map = np.nanmean(val_aps)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_map"].append(val_map)
        history["val_c1_ap"].append(val_aps[0])
        history["val_c2_ap"].append(val_aps[1])
        history["val_c3_ap"].append(val_aps[2])
        history["lr"].append(current_lr)

        improved = ""
        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            improved = " *"
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:>3}/{NUM_EPOCHS} | "
            f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
            f"mAP: {val_map:.4f} | "
            f"C1: {val_aps[0]:.3f} C2: {val_aps[1]:.3f} C3: {val_aps[2]:.3f} | "
            f"LR: {current_lr:.6f}{improved}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    print(f"  Best val mAP: {best_val_map:.4f} at epoch {best_epoch}")

    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD PREVIOUS MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def load_previous_models():
    print("\n" + "=" * 70)
    print("LOADING PREVIOUS MODELS")
    print("=" * 70)

    models = {}
    histories = {}

    # Exp1 models
    for mk in ["ann1", "ann2", "ann3", "mv"]:
        ckpt = PREV_DIR_1 / f"model_{mk}" / "best_model.pt"
        hist_path = PREV_DIR_1 / f"model_{mk}" / "training_history.csv"
        print(f"  Loading {MODEL_NAMES[mk]} from {ckpt.parent.name}...")
        model = create_model().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()
        models[mk] = model
        histories[mk] = pd.read_csv(hist_path).to_dict("list") if hist_path.exists() else None

    # Exp2 models
    for mk in ["and23", "or23"]:
        ckpt = PREV_DIR_2 / f"model_{mk}" / "best_model.pt"
        hist_path = PREV_DIR_2 / f"model_{mk}" / "training_history.csv"
        print(f"  Loading {MODEL_NAMES[mk]} from {ckpt.parent.name}...")
        model = create_model().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()
        models[mk] = model
        histories[mk] = pd.read_csv(hist_path).to_dict("list") if hist_path.exists() else None

    return models, histories


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ap(y_true, y_score):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan"), float("nan")
    ap = average_precision_score(y_true, y_score)
    rng = np.random.RandomState(42)
    boot_aps = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.randint(0, len(y_true), len(y_true))
        bt, bs = y_true[idx], y_score[idx]
        if bt.sum() == 0 or bt.sum() == len(bt):
            continue
        boot_aps.append(average_precision_score(bt, bs))
    if len(boot_aps) < 10:
        return ap, float("nan"), float("nan")
    alpha = (1 - BOOTSTRAP_CI) / 2
    return ap, np.percentile(boot_aps, alpha * 100), np.percentile(boot_aps, (1 - alpha) * 100)


def full_evaluation(models, df):
    print("\n" + "=" * 70)
    print("FULL EVALUATION")
    print("=" * 70)

    test_df = df[df["split"] == "test"]
    test_transform = get_transforms(is_train=False)

    # Get predictions from each model
    all_preds = {}
    for mk in ALL_MODELS:
        model = models[mk]
        dataset = CVSDataset(test_df, "mv", test_transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
        _, preds = evaluate_model(model, loader)
        all_preds[mk] = preds

    # Get true labels for each eval label set
    all_labels = {}
    for ls in EVAL_LABEL_SETS:
        all_labels[ls] = np.column_stack([test_df[f"{c}_{ls}"].values for c in CRITERIA])

    # Compute AP for every model x eval_label x criterion combination
    results = {}
    for mk in ALL_MODELS:
        results[mk] = {}
        for ls in EVAL_LABEL_SETS:
            results[mk][ls] = {}
            for ci, criterion in enumerate(CRITERIA):
                y_true = all_labels[ls][:, ci]
                y_score = all_preds[mk][:, ci]
                ap, lo, hi = bootstrap_ap(y_true, y_score)
                results[mk][ls][criterion] = {"ap": ap, "lo": lo, "hi": hi}

    return results, all_preds, all_labels


def print_comparison_table(results):
    """Print the main comparison table: all models on MV labels."""
    print("\n" + "=" * 70)
    print("COMPLETE COMPARISON TABLE (evaluated on MV labels)")
    print("=" * 70)

    print(f"\n{'Model':<18} {'C1 AP':>10} {'C2 AP':>10} {'C3 AP':>10} {'Mean AP':>10}")
    print("-" * 60)

    rows_for_csv = []

    for mk in ALL_MODELS:
        c1 = results[mk]["mv"]["C1"]["ap"] * 100
        c2 = results[mk]["mv"]["C2"]["ap"] * 100
        c3 = results[mk]["mv"]["C3"]["ap"] * 100
        mean = np.mean([c1, c2, c3])
        marker = ""
        if mk == "mixed":
            marker = " <--"
        print(f"{MODEL_NAMES[mk]:<18} {c1:>9.1f}% {c2:>9.1f}% {c3:>9.1f}% {mean:>9.1f}%{marker}")
        rows_for_csv.append({
            "Model": MODEL_NAMES[mk],
            "C1_AP_MV": c1, "C2_AP_MV": c2, "C3_AP_MV": c3, "Mean_AP_MV": mean,
        })

    # SwinCVS baselines
    print("-" * 60)
    for name, ref in [("SwinCVS Frozen", SWINCVS_FROZEN), ("SwinCVS E2E", SWINCVS_E2E)]:
        print(f"{name:<18} {ref['C1']:>9.1f}% {ref['C2']:>9.1f}% {ref['C3']:>9.1f}% {ref['mean']:>9.1f}%")
        rows_for_csv.append({
            "Model": name,
            "C1_AP_MV": ref["C1"], "C2_AP_MV": ref["C2"],
            "C3_AP_MV": ref["C3"], "Mean_AP_MV": ref["mean"],
        })

    return pd.DataFrame(rows_for_csv)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison_bar(results):
    """Bar chart: all models + SwinCVS on MV labels."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    models_plus = ALL_MODELS + ["swincvs_frozen", "swincvs_e2e"]
    names = [MODEL_NAMES.get(m, m) for m in ALL_MODELS] + ["SwinCVS Frozen", "SwinCVS E2E"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6",
              "#f39c12", "#1abc9c", "#e67e22",  # 7 ViT models
              "#7f8c8d", "#34495e"]  # SwinCVS

    # Gather data
    data = {}
    for mk in ALL_MODELS:
        data[mk] = {c: results[mk]["mv"][c]["ap"] * 100 for c in CRITERIA}
        data[mk]["mean"] = np.mean(list(data[mk].values()))
    data["swincvs_frozen"] = {"C1": SWINCVS_FROZEN["C1"], "C2": SWINCVS_FROZEN["C2"],
                               "C3": SWINCVS_FROZEN["C3"], "mean": SWINCVS_FROZEN["mean"]}
    data["swincvs_e2e"] = {"C1": SWINCVS_E2E["C1"], "C2": SWINCVS_E2E["C2"],
                            "C3": SWINCVS_E2E["C3"], "mean": SWINCVS_E2E["mean"]}

    for idx, metric in enumerate(["C1", "C2", "C3", "mean"]):
        ax = axes[idx]
        vals = [data[mk][metric] for mk in models_plus]
        x = np.arange(len(models_plus))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)

        # Highlight best ViT and mixed
        best_vit_idx = np.argmax(vals[:7])
        bars[best_vit_idx].set_edgecolor("red")
        bars[best_vit_idx].set_linewidth(2.5)
        # Highlight mixed
        bars[6].set_hatch("//")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
        title = f"{metric}" if metric != "mean" else "Mean"
        ax.set_title(f"{title} AP (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 85)

        for i, v in enumerate(vals):
            ax.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=7, fontweight="bold")

    plt.suptitle("All Models on Majority Vote Labels (ViT + SwinCVS Baselines)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved comparison_bar.png")


def plot_training_curves(all_histories):
    """Training curves for all 7 models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {
        "ann1": "#e74c3c", "ann2": "#3498db", "ann3": "#2ecc71",
        "mv": "#9b59b6", "and23": "#f39c12", "or23": "#1abc9c",
        "mixed": "#e67e22",
    }
    styles = {
        "ann1": "--", "ann2": "--", "ann3": "--", "mv": "--",
        "and23": "-.", "or23": "-.", "mixed": "-",
    }

    # Val mAP
    ax = axes[0]
    for mk in ALL_MODELS:
        h = all_histories.get(mk)
        if h:
            ax.plot(h["epoch"], h["val_map"], label=MODEL_NAMES[mk],
                    color=colors[mk], linewidth=2 if mk == "mixed" else 1.2,
                    linestyle=styles[mk])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Mean AP")
    ax.set_title("Validation mAP", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Train loss
    ax = axes[1]
    for mk in ALL_MODELS:
        h = all_histories.get(mk)
        if h:
            ax.plot(h["epoch"], h["train_loss"], label=MODEL_NAMES[mk],
                    color=colors[mk], linewidth=2 if mk == "mixed" else 1.2,
                    linestyle=styles[mk])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves: All 7 Models (solid=mixed, dashed=exp1, dashdot=exp2)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved training_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results, comparison_df):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    mixed = results["mixed"]["mv"]
    c1 = mixed["C1"]["ap"] * 100
    c2 = mixed["C2"]["ap"] * 100
    c3 = mixed["C3"]["ap"] * 100
    mean = np.mean([c1, c2, c3])
    expected = 49.3

    print(f"\n  Mixed Strategy Model:")
    print(f"    C1: {c1:.1f}% (target: ~50.1%, source: Ann3)")
    print(f"    C2: {c2:.1f}% (target: ~41.2%, source: MV)")
    print(f"    C3: {c3:.1f}% (target: ~56.7%, source: AND23)")
    print(f"    Mean: {mean:.1f}% (expected: ~{expected:.1f}%)")

    # 1. Does it achieve expected?
    print(f"\n  1. Does mixed achieve ~{expected:.1f}% mean AP?")
    diff = mean - expected
    print(f"     Actual: {mean:.1f}% ({diff:+.1f}% vs expected)")

    # 2. Does it beat every single-strategy model?
    print(f"\n  2. Does mixed beat every single-strategy model on mean AP?")
    for mk in ["ann1", "ann2", "ann3", "mv", "and23", "or23"]:
        mk_c1 = results[mk]["mv"]["C1"]["ap"] * 100
        mk_c2 = results[mk]["mv"]["C2"]["ap"] * 100
        mk_c3 = results[mk]["mv"]["C3"]["ap"] * 100
        mk_mean = np.mean([mk_c1, mk_c2, mk_c3])
        diff = mean - mk_mean
        marker = "YES" if diff > 0 else "NO"
        print(f"     vs {MODEL_NAMES[mk]:>15}: {mk_mean:.1f}% -> {diff:+.1f}% [{marker}]")

    # 3. Per-criterion vs best individual
    print(f"\n  3. Per-criterion vs best individual strategy:")
    best_prev = {
        "C1": ("ann3", results["ann3"]["mv"]["C1"]["ap"] * 100),
        "C2": ("mv", results["mv"]["mv"]["C2"]["ap"] * 100),
        "C3": ("and23", results["and23"]["mv"]["C3"]["ap"] * 100),
    }
    for criterion in CRITERIA:
        mixed_ap = results["mixed"]["mv"][criterion]["ap"] * 100
        best_mk, best_ap = best_prev[criterion]
        diff = mixed_ap - best_ap
        marker = "MATCH/BEAT" if diff >= -0.5 else "BELOW"
        print(f"     {criterion}: Mixed={mixed_ap:.1f}% vs Best({MODEL_NAMES[best_mk]})={best_ap:.1f}% "
              f"({diff:+.1f}%) [{marker}]")

    # 4. Gap to SwinCVS
    print(f"\n  4. Gap to SwinCVS:")
    for name, ref in [("SwinCVS Frozen", SWINCVS_FROZEN), ("SwinCVS E2E", SWINCVS_E2E)]:
        gap = mean - ref["mean"]
        print(f"     vs {name}: {ref['mean']:.1f}% -> gap = {gap:+.1f}%")
        for criterion in CRITERIA:
            mixed_ap = results["mixed"]["mv"][criterion]["ap"] * 100
            ref_ap = ref[criterion]
            print(f"       {criterion}: {mixed_ap:.1f}% vs {ref_ap:.1f}% ({mixed_ap - ref_ap:+.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print(f"Criterion-Specific Label Strategy Experiment")
    print(f"Started: {datetime.datetime.now()}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    df = load_and_prepare_data()

    # Pre-flight
    run_preflight(df)

    # Load previous models
    models, all_histories = load_previous_models()

    # Train mixed model
    mixed_model, mixed_history = train_mixed_model(df)
    models["mixed"] = mixed_model
    all_histories["mixed"] = mixed_history

    # Full evaluation
    results, all_preds, all_labels = full_evaluation(models, df)

    # Comparison table
    comparison_df = print_comparison_table(results)
    comparison_df.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
    print("  Saved comparison_table.csv")

    # Visualisations
    print("\n" + "=" * 70)
    print("GENERATING VISUALISATIONS")
    print("=" * 70)
    plot_comparison_bar(results)
    plot_training_curves(all_histories)

    # Summary
    print_summary(results, comparison_df)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
