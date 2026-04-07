"""
Annotator 2+3 Label Strategies Experiment
==========================================
Tests whether combining Annotator 2+3 labels (AND/OR) outperforms
majority vote and single-annotator labels for CVS classification.

Trains 2 new models (AND, OR) and evaluates all 6 models against all 6
label sets, producing 6x6 AP matrices per criterion.
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
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support,
)
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
OUTPUT_DIR = BASE_DIR / "annotator_experiment" / "outputs_exp2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREV_OUTPUT_DIR = BASE_DIR / "annotator_experiment" / "outputs"

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

# All 6 label sets
LABEL_SETS = ["ann1", "ann2", "ann3", "mv", "and23", "or23"]
LABEL_SET_NAMES = {
    "ann1": "Annotator 1",
    "ann2": "Annotator 2",
    "ann3": "Annotator 3",
    "mv": "Majority Vote",
    "and23": "Ann2+3 AND",
    "or23": "Ann2+3 OR",
}

# Models: 4 previous + 2 new
ALL_MODELS = ["ann1", "ann2", "ann3", "mv", "and23", "or23"]
NEW_MODELS = ["and23", "or23"]
PREV_MODELS = ["ann1", "ann2", "ann3", "mv"]


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
    if vid <= 120:
        return "train"
    elif vid <= 161:
        return "val"
    else:
        return "test"


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
            fsize = img_path.stat().st_size
            if fsize < 10000:
                img = Image.open(img_path)
                arr = np.array(img)
                if arr.mean() == 0:
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

    # AND strategy: both Ann2 AND Ann3 say yes
    for c in CRITERIA:
        df[f"{c}_and23"] = ((df[f"{c}_ann2"] == 1) & (df[f"{c}_ann3"] == 1)).astype(float)

    # OR strategy: Ann2 OR Ann3 says yes
    for c in CRITERIA:
        df[f"{c}_or23"] = ((df[f"{c}_ann2"] == 1) | (df[f"{c}_ann3"] == 1)).astype(float)

    # Drop missing annotator data
    ann_cols = [f"{c}_ann{a}" for c in CRITERIA for a in [1, 2, 3]]
    df = df.dropna(subset=ann_cols)

    # Assign splits
    df["split"] = df["vid"].apply(get_split)

    # Exclude black frames
    black_frames = identify_black_frames(df)
    before = len(df)
    df = df[~df.apply(lambda r: (int(r["vid"]), int(r["frame"])) in black_frames, axis=1)]
    print(f"After excluding black frames: {len(df)} ({before - len(df)} excluded)")

    # Build image paths
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
    print("PRE-FLIGHT: POSITIVE RATES COMPARISON")
    print("=" * 70)

    print(f"\n{'Label Strategy':<18} {'C1+ rate':>10} {'C2+ rate':>10} {'C3+ rate':>10}")
    print("-" * 50)
    for ls in LABEL_SETS:
        name = LABEL_SET_NAMES[ls]
        rates = []
        for c in CRITERIA:
            col = f"{c}_{ls}"
            rate = df[col].mean() * 100
            rates.append(rate)
        print(f"{name:<18} {rates[0]:>9.1f}% {rates[1]:>9.1f}% {rates[2]:>9.1f}%")

    # Per-split breakdown for new strategies
    print("\n  Per-split positive rates for new strategies:")
    for ls in NEW_MODELS:
        print(f"\n  {LABEL_SET_NAMES[ls]}:")
        for split in ["train", "val", "test"]:
            sdf = df[df["split"] == split]
            rates = [sdf[f"{c}_{ls}"].mean() * 100 for c in CRITERIA]
            print(f"    {split}: C1={rates[0]:.1f}%  C2={rates[1]:.1f}%  C3={rates[2]:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET & MODEL (same as exp1)
# ═══════════════════════════════════════════════════════════════════════════════

class CVSDataset(Dataset):
    def __init__(self, df, label_set, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_set = label_set
        self.transform = transform
        self.label_cols = [f"{c}_{label_set}" for c in CRITERIA]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(
            [row[col] for col in self.label_cols], dtype=torch.float32
        )
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
        w = neg / max(pos, 1)
        weights.append(w)
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

def train_one_model(df, label_set, model_idx, total):
    name = LABEL_SET_NAMES[label_set]
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL {model_idx}/{total}: {name}")
    print(f"{'='*70}")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_dataset = CVSDataset(train_df, label_set, get_transforms(is_train=True))
    val_dataset = CVSDataset(val_df, label_set, get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = create_model().to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    pos_weight = compute_pos_weights(train_df, label_set).to(DEVICE)
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

    save_dir = OUTPUT_DIR / f"model_{label_set}"
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

        # Validate
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
            y_true = val_labels[:, i]
            y_score = val_preds[:, i]
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

def load_previous_model(label_set):
    """Load a model checkpoint from the previous experiment."""
    ckpt_path = PREV_OUTPUT_DIR / f"model_{label_set}" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Previous checkpoint not found: {ckpt_path}")
    model = create_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    return model


def load_previous_history(label_set):
    """Load training history from the previous experiment."""
    csv_path = PREV_OUTPUT_DIR / f"model_{label_set}" / "training_history.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path).to_dict(orient="list")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ap(y_true, y_score, n_bootstrap=BOOTSTRAP_N, ci=BOOTSTRAP_CI):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan"), float("nan")
    ap = average_precision_score(y_true, y_score)
    rng = np.random.RandomState(42)
    boot_aps = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), len(y_true))
        bt, bs = y_true[idx], y_score[idx]
        if bt.sum() == 0 or bt.sum() == len(bt):
            continue
        boot_aps.append(average_precision_score(bt, bs))
    if len(boot_aps) < 10:
        return ap, float("nan"), float("nan")
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_aps, alpha * 100)
    hi = np.percentile(boot_aps, (1 - alpha) * 100)
    return ap, lo, hi


def full_evaluation(models, df):
    """Evaluate all 6 models against all 6 label sets."""
    print("\n" + "=" * 70)
    print("FULL EVALUATION (6x6 matrices)")
    print("=" * 70)

    test_df = df[df["split"] == "test"]
    test_transform = get_transforms(is_train=False)

    # Get predictions from each model
    all_preds = {}
    for model_key in ALL_MODELS:
        model = models[model_key]
        dataset = CVSDataset(test_df, "mv", test_transform)  # labels don't matter for preds
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
        _, preds = evaluate_model(model, loader)
        all_preds[model_key] = preds

    # Get true labels for each label set
    all_labels = {}
    for ls in LABEL_SETS:
        labels = np.column_stack([test_df[f"{c}_{ls}"].values for c in CRITERIA])
        all_labels[ls] = labels

    # Build 6x6 AP matrices
    results = {}
    for ci, criterion in enumerate(CRITERIA):
        n = len(ALL_MODELS)
        matrix = np.zeros((n, n))
        matrix_lo = np.zeros((n, n))
        matrix_hi = np.zeros((n, n))

        for ri, train_key in enumerate(ALL_MODELS):
            for cj, eval_key in enumerate(LABEL_SETS):
                y_true = all_labels[eval_key][:, ci]
                y_score = all_preds[train_key][:, ci]
                ap, lo, hi = bootstrap_ap(y_true, y_score)
                matrix[ri, cj] = ap
                matrix_lo[ri, cj] = lo
                matrix_hi[ri, cj] = hi

        results[criterion] = {"ap": matrix, "lo": matrix_lo, "hi": matrix_hi}

        # Print matrix
        print(f"\n  {criterion} AP Matrix (rows=trained on, cols=evaluated against):")
        header = f"  {'':>15}"
        for ls in LABEL_SETS:
            header += f" {LABEL_SET_NAMES[ls]:>13}"
        print(header)
        print(f"  {'-'*99}")
        for ri, mk in enumerate(ALL_MODELS):
            row = f"  {LABEL_SET_NAMES[mk]:>15}"
            for cj in range(n):
                ap = matrix[ri, cj]
                row += f"  {ap*100:>5.1f} [{matrix_lo[ri,cj]*100:.0f}-{matrix_hi[ri,cj]*100:.0f}]"
            print(row)

    # Secondary metrics
    secondary = {}
    for train_key in ALL_MODELS:
        for eval_key in LABEL_SETS:
            key = (train_key, eval_key)
            preds = all_preds[train_key]
            labels = all_labels[eval_key]
            row = {}
            for ci, criterion in enumerate(CRITERIA):
                y_true = labels[:, ci]
                y_score = preds[:, ci]
                if y_true.sum() == 0 or y_true.sum() == len(y_true):
                    row[criterion] = {"auroc": np.nan, "precision": np.nan,
                                      "recall": np.nan, "f1": np.nan}
                    continue
                auroc = roc_auc_score(y_true, y_score)
                thresholds = np.linspace(0, 1, 101)
                best_f1, best_t = 0, 0.5
                for t in thresholds:
                    y_pred = (y_score >= t).astype(int)
                    p, r, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average="binary", zero_division=0
                    )
                    if f1 > best_f1:
                        best_f1, best_t = f1, t
                y_pred = (y_score >= best_t).astype(int)
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0
                )
                row[criterion] = {"auroc": auroc, "precision": p, "recall": r,
                                  "f1": f1, "threshold": best_t}
            secondary[key] = row

    return results, secondary, all_preds, all_labels


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmaps(results):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    labels = [LABEL_SET_NAMES[ls] for ls in LABEL_SETS]
    model_labels = [LABEL_SET_NAMES[mk] for mk in ALL_MODELS]

    for idx, criterion in enumerate(CRITERIA):
        ax = axes[idx]
        matrix = results[criterion]["ap"] * 100
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=80, aspect="auto")

        for i in range(6):
            for j in range(6):
                val = matrix[i, j]
                color = "white" if val > 55 else "black"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color=color, fontweight=weight, fontsize=9)

        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(model_labels, fontsize=8)
        ax.set_xlabel("Evaluated Against", fontsize=10)
        ax.set_ylabel("Trained On", fontsize=10)
        ax.set_title(f"{criterion} AP (%)", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=axes, shrink=0.8, label="AP (%)")
    plt.suptitle("6x6 AP Matrices: All Label Strategies", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmaps_6x6.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved heatmaps_6x6.png")


def plot_mv_comparison(results):
    """Bar chart: all 6 models evaluated on MV labels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    mv_idx = LABEL_SETS.index("mv")

    for idx, criterion in enumerate(CRITERIA):
        ax = axes[idx]
        matrix = results[criterion]["ap"] * 100
        lo_matrix = results[criterion]["lo"] * 100
        hi_matrix = results[criterion]["hi"] * 100

        vals = matrix[:, mv_idx]
        errs_lo = vals - lo_matrix[:, mv_idx]
        errs_hi = hi_matrix[:, mv_idx] - vals
        errs = np.array([errs_lo, errs_hi])

        x = np.arange(6)
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor="black", capsize=4)

        # Highlight best
        best_idx = np.nanargmax(vals)
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels([LABEL_SET_NAMES[mk] for mk in ALL_MODELS],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("AP (%)")
        ax.set_title(f"{criterion}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 70)

        # Add value labels
        for i, v in enumerate(vals):
            ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("All Models Evaluated on Majority Vote Labels (with 95% CI)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mv_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved mv_comparison.png")


def plot_training_curves(all_histories):
    """Overlay training curves for all 6 models."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {
        "ann1": "#e74c3c", "ann2": "#3498db", "ann3": "#2ecc71",
        "mv": "#9b59b6", "and23": "#f39c12", "or23": "#1abc9c",
    }

    # Val mAP
    ax = axes[0]
    for mk in ALL_MODELS:
        h = all_histories[mk]
        if h is not None:
            ax.plot(h["epoch"], h["val_map"], label=LABEL_SET_NAMES[mk],
                    color=colors[mk], linewidth=1.5,
                    linestyle="--" if mk in PREV_MODELS else "-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Mean AP")
    ax.set_title("Validation mAP", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Train loss
    ax = axes[1]
    for mk in ALL_MODELS:
        h = all_histories[mk]
        if h is not None:
            ax.plot(h["epoch"], h["train_loss"], label=LABEL_SET_NAMES[mk],
                    color=colors[mk], linewidth=1.5,
                    linestyle="--" if mk in PREV_MODELS else "-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Val loss
    ax = axes[2]
    for mk in ALL_MODELS:
        h = all_histories[mk]
        if h is not None:
            ax.plot(h["epoch"], h["val_loss"], label=LABEL_SET_NAMES[mk],
                    color=colors[mk], linewidth=1.5,
                    linestyle="--" if mk in PREV_MODELS else "-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves: All 6 Models (dashed=previous, solid=new)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves_all.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved training_curves_all.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE & SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(results, secondary):
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save AP matrices
    for criterion in CRITERIA:
        matrix = results[criterion]["ap"] * 100
        matrix_df = pd.DataFrame(
            matrix,
            index=[LABEL_SET_NAMES[mk] for mk in ALL_MODELS],
            columns=[LABEL_SET_NAMES[ls] for ls in LABEL_SETS],
        )
        matrix_df.to_csv(OUTPUT_DIR / f"ap_matrix_{criterion}.csv")

    # Save detailed results
    rows = []
    for train_key in ALL_MODELS:
        for eval_key in LABEL_SETS:
            for ci, criterion in enumerate(CRITERIA):
                ri = ALL_MODELS.index(train_key)
                cj = LABEL_SETS.index(eval_key)
                ap = results[criterion]["ap"][ri, cj]
                lo = results[criterion]["lo"][ri, cj]
                hi = results[criterion]["hi"][ri, cj]
                sec = secondary.get((train_key, eval_key), {}).get(criterion, {})
                rows.append({
                    "trained_on": LABEL_SET_NAMES[train_key],
                    "evaluated_on": LABEL_SET_NAMES[eval_key],
                    "criterion": criterion,
                    "AP": ap, "AP_CI_lo": lo, "AP_CI_hi": hi,
                    "AUROC": sec.get("auroc", np.nan),
                    "Precision": sec.get("precision", np.nan),
                    "Recall": sec.get("recall", np.nan),
                    "F1": sec.get("f1", np.nan),
                    "Threshold": sec.get("threshold", np.nan),
                })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "detailed_results.csv", index=False)
    print("  Saved detailed_results.csv")


def print_summary(results):
    print("\n" + "=" * 70)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 70)

    mv_idx = LABEL_SETS.index("mv")

    # Q1: Do AND/OR beat MV on MV labels?
    print("\n1. Do AND/OR models beat MV model on MV labels?")
    mv_model_idx = ALL_MODELS.index("mv")
    and_model_idx = ALL_MODELS.index("and23")
    or_model_idx = ALL_MODELS.index("or23")

    for ci, criterion in enumerate(CRITERIA):
        mv_ap = results[criterion]["ap"][mv_model_idx, mv_idx] * 100
        and_ap = results[criterion]["ap"][and_model_idx, mv_idx] * 100
        or_ap = results[criterion]["ap"][or_model_idx, mv_idx] * 100
        print(f"   {criterion}: MV={mv_ap:.1f}%  AND={and_ap:.1f}% ({and_ap-mv_ap:+.1f})  "
              f"OR={or_ap:.1f}% ({or_ap-mv_ap:+.1f})")

    # Q2: Do they beat the best single-annotator model?
    print("\n2. Do AND/OR beat the best single-annotator model?")
    best_prev = {"C1": "ann3", "C2": "mv", "C3": "ann2"}  # from exp1
    for ci, criterion in enumerate(CRITERIA):
        best_key = best_prev[criterion]
        best_idx = ALL_MODELS.index(best_key)
        best_ap = results[criterion]["ap"][best_idx, mv_idx] * 100
        and_ap = results[criterion]["ap"][and_model_idx, mv_idx] * 100
        or_ap = results[criterion]["ap"][or_model_idx, mv_idx] * 100
        winner = "AND" if and_ap > or_ap else "OR"
        winner_ap = max(and_ap, or_ap)
        diff = winner_ap - best_ap
        marker = ">>>" if diff > 0 else "   "
        print(f"   {marker} {criterion}: Best prev ({LABEL_SET_NAMES[best_key]})={best_ap:.1f}%  "
              f"Best new ({winner})={winner_ap:.1f}% ({diff:+.1f}%)")

    # Q3: Self-AP (diagonal)
    print("\n3. Self-AP (diagonal) for all strategies:")
    for ci, criterion in enumerate(CRITERIA):
        print(f"   {criterion}:")
        for ri, mk in enumerate(ALL_MODELS):
            self_ap = results[criterion]["ap"][ri, ri] * 100
            print(f"     {LABEL_SET_NAMES[mk]:>15}: {self_ap:.1f}%")

    # Q4: Best strategy per criterion on MV
    print("\n4. BEST LABEL STRATEGY per criterion (evaluated on MV):")
    for ci, criterion in enumerate(CRITERIA):
        aps = []
        for ri, mk in enumerate(ALL_MODELS):
            ap = results[criterion]["ap"][ri, mv_idx] * 100
            aps.append((mk, ap))
        aps.sort(key=lambda x: x[1], reverse=True)
        print(f"   {criterion}: ", end="")
        for rank, (mk, ap) in enumerate(aps):
            marker = " <-- BEST" if rank == 0 else ""
            print(f"{LABEL_SET_NAMES[mk]}={ap:.1f}%{marker}", end="  ")
        print()

    # Overall recommendation
    print("\n5. OVERALL RECOMMENDATION:")
    overall_scores = {}
    for ri, mk in enumerate(ALL_MODELS):
        mean_ap = np.mean([results[c]["ap"][ri, mv_idx] for c in CRITERIA]) * 100
        overall_scores[mk] = mean_ap
    best_mk = max(overall_scores, key=overall_scores.get)
    for mk in ALL_MODELS:
        marker = " <<<" if mk == best_mk else ""
        print(f"   {LABEL_SET_NAMES[mk]:>15}: mean AP = {overall_scores[mk]:.1f}%{marker}")
    print(f"\n   Recommended: {LABEL_SET_NAMES[best_mk]} (mean AP: {overall_scores[best_mk]:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print(f"Ann2+3 Label Strategy Experiment")
    print(f"Started: {datetime.datetime.now()}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    df = load_and_prepare_data()

    # Pre-flight
    run_preflight(df)

    # Load previous models
    print("\n" + "=" * 70)
    print("LOADING PREVIOUS MODELS")
    print("=" * 70)
    models = {}
    all_histories = {}
    for mk in PREV_MODELS:
        print(f"  Loading {LABEL_SET_NAMES[mk]}...")
        models[mk] = load_previous_model(mk)
        all_histories[mk] = load_previous_history(mk)

    # Train new models
    for idx, mk in enumerate(NEW_MODELS, 1):
        model, history = train_one_model(df, mk, idx, len(NEW_MODELS))
        models[mk] = model
        all_histories[mk] = history

    # Full evaluation
    results, secondary, all_preds, all_labels = full_evaluation(models, df)

    # Visualisations
    print("\n" + "=" * 70)
    print("GENERATING VISUALISATIONS")
    print("=" * 70)
    plot_heatmaps(results)
    plot_mv_comparison(results)
    plot_training_curves(all_histories)

    # Save
    save_results(results, secondary)

    # Summary
    print_summary(results)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
