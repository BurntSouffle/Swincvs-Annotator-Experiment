"""
Annotator-Specific Training Experiment
=======================================
Tests whether training on consistent single-annotator labels outperforms
training on noisy majority-vote labels for CVS classification.

Trains 4 ViT-Base models (one per annotator + majority vote) and evaluates
each against all 4 label sets, producing 4x4 AP matrices per criterion.
"""

import os
import sys
import ast
import json
import time
import datetime
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support, cohen_kappa_score
)
from tqdm import tqdm
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("C:/Users/sufia/Documents/Uni/Masters/DISSERTATION")
ENDOSCAPES_DIR = BASE_DIR / "endoscapes"
METADATA_CSV = ENDOSCAPES_DIR / "all_metadata.csv"
OUTPUT_DIR = BASE_DIR / "annotator_experiment" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
LABEL_SETS = ["ann1", "ann2", "ann3", "mv"]
LABEL_SET_NAMES = {
    "ann1": "Annotator 1",
    "ann2": "Annotator 2",
    "ann3": "Annotator 3",
    "mv": "Majority Vote",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & LABEL PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_annotator_votes(vote_str):
    """Parse '[0, 1, 0]' string into list of ints."""
    try:
        parsed = ast.literal_eval(vote_str)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
            return [int(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    return None


def identify_black_frames(df):
    """Identify black/corrupted frames (mean pixel intensity = 0).

    Uses file size as a fast pre-filter: black JPEGs are very small (<1KB).
    Only opens ambiguous files.
    """
    black_set = set()
    print("Scanning for black frames...")
    # Build all paths first for speed
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
            if fsize < 10000:  # Black JPEGs are ~7KB; real frames are >>10KB
                img = Image.open(img_path)
                arr = np.array(img)
                if arr.mean() == 0:
                    black_set.add((vid, frame))
        except Exception:
            black_set.add((vid, frame))
    print(f"  Found {len(black_set)} black/corrupted frames")
    return black_set


def get_split(vid):
    if vid <= 120:
        return "train"
    elif vid <= 161:
        return "val"
    else:
        return "test"


def load_and_prepare_data():
    """Load metadata, parse annotator labels, exclude black frames."""
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

    # Majority vote binarisation
    for c in CRITERIA:
        df[f"{c}_mv"] = (df[c] >= 0.5).astype(float)

    # Drop rows with missing annotator data
    ann_cols = [f"{c}_ann{a}" for c in CRITERIA for a in [1, 2, 3]]
    before = len(df)
    df = df.dropna(subset=ann_cols)
    print(f"After dropping missing annotator data: {len(df)} ({before - len(df)} dropped)")

    # Assign splits
    df["split"] = df["vid"].apply(get_split)

    # Identify and exclude black frames
    black_frames = identify_black_frames(df)
    before = len(df)
    df = df[~df.apply(lambda r: (int(r["vid"]), int(r["frame"])) in black_frames, axis=1)]
    print(f"After excluding black frames: {len(df)} ({before - len(df)} excluded)")

    # Build image paths
    df["img_path"] = df.apply(
        lambda r: str(ENDOSCAPES_DIR / r["split"] / f"{int(r['vid'])}_{int(r['frame'])}.jpg"),
        axis=1
    )

    # Verify a sample of paths exist
    sample = df.sample(min(100, len(df)))
    missing = sum(1 for p in sample["img_path"] if not Path(p).exists())
    if missing > 0:
        print(f"  WARNING: {missing}/{len(sample)} sampled paths do not exist!")

    print(f"\nFinal dataset: {len(df)} frames")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {(df['split'] == split).sum()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def run_preflight_checks(df):
    """Compute and print all pre-flight statistics."""
    print("\n" + "=" * 70)
    print("PRE-FLIGHT CHECKS")
    print("=" * 70)

    # 1. Per-annotator positive rates
    print("\n-- 1. Per-Annotator Positive Rates --")
    print(f"{'Annotator':<15} {'C1+ rate':>10} {'C2+ rate':>10} {'C3+ rate':>10}")
    print("-" * 47)
    for label_set in LABEL_SETS:
        name = LABEL_SET_NAMES[label_set]
        rates = []
        for c in CRITERIA:
            col = f"{c}_{label_set}"
            rate = df[col].mean() * 100
            rates.append(rate)
        print(f"{name:<15} {rates[0]:>9.1f}% {rates[1]:>9.1f}% {rates[2]:>9.1f}%")

    # 2. Pairwise agreement rates
    print("\n-- 2. Pairwise Agreement Rates --")
    for c in CRITERIA:
        print(f"\n  {c}:")
        print(f"  {'Pair':<15} {'Agreement':>10}")
        print(f"  {'-'*27}")
        for i in range(1, 4):
            for j in range(i + 1, 4):
                col_i = f"{c}_ann{i}"
                col_j = f"{c}_ann{j}"
                agree = (df[col_i] == df[col_j]).mean() * 100
                print(f"  Ann{i} vs Ann{j}    {agree:>9.1f}%")

    # 3. Cohen's kappa
    print("\n-- 3. Cohen's Kappa --")
    for c in CRITERIA:
        print(f"\n  {c}:")
        print(f"  {'Pair':<15} {'Kappa':>10}")
        print(f"  {'-'*27}")
        for i in range(1, 4):
            for j in range(i + 1, 4):
                col_i = f"{c}_ann{i}"
                col_j = f"{c}_ann{j}"
                kappa = cohen_kappa_score(df[col_i].astype(int), df[col_j].astype(int))
                print(f"  Ann{i} vs Ann{j}    {kappa:>10.4f}")

    # 4. Frame counts per split
    print("\n-- 4. Frame Counts Per Split --")
    for split in ["train", "val", "test"]:
        count = (df["split"] == split).sum()
        print(f"  {split}: {count}")
    print(f"  Total: {len(df)}")

    # 5. Class distribution per criterion per annotator per split
    print("\n-- 5. Class Distribution Per Split --")
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        print(f"\n  {split.upper()} ({len(split_df)} frames):")
        header = f"  {'Label Set':<15}"
        for c in CRITERIA:
            header += f" {c}+ count ({c}+ %)"
        print(header)
        print(f"  {'-'*75}")
        for label_set in LABEL_SETS:
            row = f"  {LABEL_SET_NAMES[label_set]:<15}"
            for c in CRITERIA:
                col = f"{c}_{label_set}"
                pos = int(split_df[col].sum())
                pct = split_df[col].mean() * 100
                row += f"    {pos:>5} ({pct:>5.1f}%)  "
            print(row)

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class CVSDataset(Dataset):
    """Dataset for CVS classification with configurable label set."""

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


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def create_model():
    """Create ViT-Base/16 with frozen layers except last 2 blocks + head."""
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 transformer blocks
    num_blocks = len(model.blocks)
    for i in range(num_blocks - 2, num_blocks):
        for param in model.blocks[i].parameters():
            param.requires_grad = True

    # Unfreeze norm layer
    if hasattr(model, "norm"):
        for param in model.norm.parameters():
            param.requires_grad = True

    # Add classification head (768 -> 3)
    head = nn.Linear(model.num_features, 3)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)

    full_model = nn.Sequential(model, head)
    return full_model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pos_weights(df, label_set):
    """Compute pos_weight for BCEWithLogitsLoss based on class balance."""
    weights = []
    for c in CRITERIA:
        col = f"{c}_{label_set}"
        pos = df[col].sum()
        neg = len(df) - pos
        w = neg / max(pos, 1)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(labels, preds):
    """Compute AP and AUROC per criterion."""
    aps, aurocs = [], []
    for i in range(3):
        y_true = labels[:, i]
        y_score = preds[:, i]
        # Need both classes present
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            aps.append(float("nan"))
            aurocs.append(float("nan"))
        else:
            aps.append(average_precision_score(y_true, y_score))
            aurocs.append(roc_auc_score(y_true, y_score))
    return aps, aurocs


def evaluate_model(model, dataloader):
    """Run inference, return labels and sigmoid predictions."""
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


def train_one_model(df, label_set, model_idx):
    """Train a single model on a given label set."""
    name = LABEL_SET_NAMES[label_set]
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL {model_idx}/4: {name}")
    print(f"{'='*70}")

    # Create datasets
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_dataset = CVSDataset(train_df, label_set, get_transforms(is_train=True))
    val_dataset = CVSDataset(val_df, label_set, get_transforms(is_train=False))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Model
    model = create_model().to(DEVICE)
    total_p, train_p = count_params(model)
    print(f"  Parameters: {total_p:,} total, {train_p:,} trainable")

    # Loss with pos_weight from training labels
    pos_weight = compute_pos_weights(train_df, label_set).to(DEVICE)
    print(f"  Pos weights: C1={pos_weight[0]:.2f}, C2={pos_weight[1]:.2f}, C3={pos_weight[2]:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimiser & scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Training loop
    best_val_map = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_map": [],
               "val_c1_ap": [], "val_c2_ap": [], "val_c3_ap": [], "lr": []}

    save_dir = OUTPUT_DIR / f"model_{label_set}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
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

        # Compute val loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses)

        # Metrics (evaluated against own label set)
        val_aps, _ = compute_metrics(val_labels, val_preds)
        val_map = np.nanmean(val_aps)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Log
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
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val mAP: {val_map:.4f} | "
            f"C1: {val_aps[0]:.3f} C2: {val_aps[1]:.3f} C3: {val_aps[2]:.3f} | "
            f"LR: {current_lr:.6f}{improved}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Save history
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    print(f"  Best val mAP: {best_val_map:.4f} at epoch {best_epoch}")

    # Load best model for return
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ap(y_true, y_score, n_bootstrap=BOOTSTRAP_N, ci=BOOTSTRAP_CI):
    """Compute AP with bootstrap confidence interval."""
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
    """Evaluate all models against all label sets. Returns 4x4 AP matrices."""
    print("\n" + "=" * 70)
    print("FULL EVALUATION (4x4 matrices)")
    print("=" * 70)

    test_df = df[df["split"] == "test"]
    test_transform = get_transforms(is_train=False)

    # Get predictions from each model (predictions are label-independent)
    all_preds = {}
    for label_set in LABEL_SETS:
        model = models[label_set]
        dataset = CVSDataset(test_df, "mv", test_transform)  # labels don't matter for preds
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        _, preds = evaluate_model(model, loader)
        all_preds[label_set] = preds

    # Get true labels for each label set
    all_labels = {}
    for label_set in LABEL_SETS:
        labels = np.column_stack([
            test_df[f"{c}_{label_set}"].values for c in CRITERIA
        ])
        all_labels[label_set] = labels

    # Build 4x4 AP matrices (one per criterion)
    results = {}
    for ci, criterion in enumerate(CRITERIA):
        matrix = np.zeros((4, 4))
        matrix_lo = np.zeros((4, 4))
        matrix_hi = np.zeros((4, 4))

        for ri, train_ls in enumerate(LABEL_SETS):
            for cj, eval_ls in enumerate(LABEL_SETS):
                y_true = all_labels[eval_ls][:, ci]
                y_score = all_preds[train_ls][:, ci]
                ap, lo, hi = bootstrap_ap(y_true, y_score)
                matrix[ri, cj] = ap
                matrix_lo[ri, cj] = lo
                matrix_hi[ri, cj] = hi

        results[criterion] = {
            "ap": matrix, "lo": matrix_lo, "hi": matrix_hi
        }

        # Print matrix
        print(f"\n  {criterion} AP Matrix (trained on rows, evaluated on columns):")
        header = f"  {'':>15}"
        for ls in LABEL_SETS:
            header += f" {LABEL_SET_NAMES[ls]:>15}"
        print(header)
        print(f"  {'-'*79}")
        for ri, train_ls in enumerate(LABEL_SETS):
            row = f"  {LABEL_SET_NAMES[train_ls]:>15}"
            for cj in range(4):
                ap = matrix[ri, cj]
                lo = matrix_lo[ri, cj]
                hi = matrix_hi[ri, cj]
                row += f"  {ap*100:>5.1f} [{lo*100:.0f}-{hi*100:.0f}]"
            print(row)

    # Secondary metrics: AUROC, precision, recall, F1
    secondary = {}
    for train_ls in LABEL_SETS:
        for eval_ls in LABEL_SETS:
            key = (train_ls, eval_ls)
            preds = all_preds[train_ls]
            labels = all_labels[eval_ls]
            row = {}
            for ci, criterion in enumerate(CRITERIA):
                y_true = labels[:, ci]
                y_score = preds[:, ci]
                if y_true.sum() == 0 or y_true.sum() == len(y_true):
                    row[criterion] = {"auroc": np.nan, "precision": np.nan,
                                      "recall": np.nan, "f1": np.nan}
                    continue
                auroc = roc_auc_score(y_true, y_score)
                # Optimal threshold on test set (for reporting only)
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
# DISAGREEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def disagreement_analysis(df, all_preds, all_labels):
    """Analyse model predictions on frames where annotators disagree on C2."""
    print("\n" + "=" * 70)
    print("DISAGREEMENT ANALYSIS (C2)")
    print("=" * 70)

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    c2_idx = 1  # C2 is index 1

    # Get per-annotator labels
    ann1 = all_labels["ann1"][:, c2_idx]
    ann2 = all_labels["ann2"][:, c2_idx]
    ann3 = all_labels["ann3"][:, c2_idx]
    mv = all_labels["mv"][:, c2_idx]

    # Find disagreement frames (not all annotators agree)
    disagree_mask = ~((ann1 == ann2) & (ann2 == ann3))
    n_disagree = disagree_mask.sum()
    print(f"\n  C2 disagreement frames: {n_disagree}/{len(test_df)} ({n_disagree/len(test_df)*100:.1f}%)")

    if n_disagree == 0:
        print("  No disagreement frames found.")
        return {}

    # Pattern: Ann1=NO, Ann2+Ann3=YES
    pattern_mask = (ann1 == 0) & (ann2 == 1) & (ann3 == 1)
    n_pattern = pattern_mask.sum()
    print(f"  Pattern Ann1=NO, Ann2+Ann3=YES: {n_pattern} frames")

    # For each disagreement frame, what does each model predict?
    disagree_results = {"frame_idx": [], "ann1_label": [], "ann2_label": [],
                        "ann3_label": [], "mv_label": []}
    for ls in LABEL_SETS:
        disagree_results[f"pred_{ls}"] = []

    for i in range(len(test_df)):
        if not disagree_mask[i]:
            continue
        disagree_results["frame_idx"].append(i)
        disagree_results["ann1_label"].append(int(ann1[i]))
        disagree_results["ann2_label"].append(int(ann2[i]))
        disagree_results["ann3_label"].append(int(ann3[i]))
        disagree_results["mv_label"].append(int(mv[i]))
        for ls in LABEL_SETS:
            disagree_results[f"pred_{ls}"].append(all_preds[ls][i, c2_idx])

    disagree_df = pd.DataFrame(disagree_results)

    # Average prediction across all 4 models
    pred_cols = [f"pred_{ls}" for ls in LABEL_SETS]
    disagree_df["avg_pred"] = disagree_df[pred_cols].mean(axis=1)
    disagree_df["model_consensus"] = (disagree_df["avg_pred"] >= 0.5).astype(int)

    # Stats
    print(f"\n  On disagreement frames:")
    for ls in LABEL_SETS:
        mean_pred = disagree_df[f"pred_{ls}"].mean()
        pred_pos = (disagree_df[f"pred_{ls}"] >= 0.5).sum()
        print(f"    Model_{ls} mean prediction: {mean_pred:.3f}, predicts positive: {pred_pos}/{n_disagree}")

    consensus_yes = disagree_df["model_consensus"].sum()
    print(f"    Model consensus (avg >= 0.5): {consensus_yes}/{n_disagree} frames predicted positive")

    # For the Ann1=NO, Ann2+Ann3=YES pattern specifically
    if n_pattern > 0:
        pattern_df = disagree_df[
            (disagree_df["ann1_label"] == 0) &
            (disagree_df["ann2_label"] == 1) &
            (disagree_df["ann3_label"] == 1)
        ]
        print(f"\n  For Ann1=NO, Ann2+Ann3=YES pattern ({n_pattern} frames):")
        for ls in LABEL_SETS:
            mean_p = pattern_df[f"pred_{ls}"].mean()
            pos = (pattern_df[f"pred_{ls}"] >= 0.5).sum()
            print(f"    Model_{ls}: mean={mean_p:.3f}, predicts YES: {pos}/{n_pattern}")

    return {"disagree_df": disagree_df, "pattern_mask": pattern_mask,
            "disagree_mask": disagree_mask}


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmaps(results):
    """Plot 4x4 AP heatmaps, one per criterion."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = [LABEL_SET_NAMES[ls] for ls in LABEL_SETS]

    for idx, criterion in enumerate(CRITERIA):
        ax = axes[idx]
        matrix = results[criterion]["ap"] * 100

        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")

        # Annotate cells
        for i in range(4):
            for j in range(4):
                val = matrix[i, j]
                color = "white" if val > 60 else "black"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color=color, fontweight=weight, fontsize=11)

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Evaluated Against", fontsize=10)
        ax.set_ylabel("Trained On", fontsize=10)
        ax.set_title(f"{criterion} AP (%)", fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=axes, shrink=0.8, label="AP (%)")
    plt.suptitle("Annotator-Specific Training: 4x4 AP Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmaps_4x4.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved heatmaps_4x4.png")


def plot_bar_charts(results):
    """Per-criterion AP bar charts - grouped by evaluation label set."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for idx, criterion in enumerate(CRITERIA):
        ax = axes[idx]
        matrix = results[criterion]["ap"] * 100
        x = np.arange(4)
        width = 0.18

        for mi, train_ls in enumerate(LABEL_SETS):
            bars = matrix[mi, :]
            ax.bar(x + mi * width - 1.5 * width, bars, width,
                   label=f"Trained: {LABEL_SET_NAMES[train_ls]}", color=colors[mi])

        ax.set_xticks(x)
        ax.set_xticklabels([LABEL_SET_NAMES[ls] for ls in LABEL_SETS],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("AP (%)")
        ax.set_title(f"{criterion}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    plt.suptitle("Per-Criterion AP by Evaluation Label Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bar_charts.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved bar_charts.png")


def plot_disagreement_scatter(disagree_info):
    """Scatter plot of Model_Ann1 vs Model_Ann3 predictions on C2 disagreement frames."""
    if not disagree_info or "disagree_df" not in disagree_info:
        return
    ddf = disagree_info["disagree_df"]
    if len(ddf) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ddf["mv_label"].map({0: "#e74c3c", 1: "#2ecc71"})
    ax.scatter(ddf["pred_ann1"], ddf["pred_ann3"], c=colors, alpha=0.6, edgecolors="k", linewidths=0.3, s=50)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Model_Ann1 Prediction (C2)", fontsize=11)
    ax.set_ylabel("Model_Ann3 Prediction (C2)", fontsize=11)
    ax.set_title("C2 Disagreement Frames: Ann1 vs Ann3 Model Predictions", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=8, label="MV = Positive"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="MV = Negative"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "disagreement_scatter_c2.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved disagreement_scatter_c2.png")


def plot_training_curves(all_histories):
    """Overlay training curves for all 4 models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"ann1": "#e74c3c", "ann2": "#3498db", "ann3": "#2ecc71", "mv": "#9b59b6"}

    # Val mAP
    ax = axes[0]
    for ls in LABEL_SETS:
        h = all_histories[ls]
        ax.plot(h["epoch"], h["val_map"], label=LABEL_SET_NAMES[ls], color=colors[ls], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Mean AP")
    ax.set_title("Validation mAP", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Train loss
    ax = axes[1]
    for ls in LABEL_SETS:
        h = all_histories[ls]
        ax.plot(h["epoch"], h["train_loss"], label=LABEL_SET_NAMES[ls], color=colors[ls], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Val loss
    ax = axes[2]
    for ls in LABEL_SETS:
        h = all_histories[ls]
        ax.plot(h["epoch"], h["val_loss"], label=LABEL_SET_NAMES[ls], color=colors[ls], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves: All 4 Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved training_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY & SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(results, secondary, disagree_info, all_histories):
    """Save all numerical results to CSV and text."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save 4x4 AP matrices as CSV
    for criterion in CRITERIA:
        matrix = results[criterion]["ap"] * 100
        matrix_df = pd.DataFrame(
            matrix,
            index=[LABEL_SET_NAMES[ls] for ls in LABEL_SETS],
            columns=[LABEL_SET_NAMES[ls] for ls in LABEL_SETS],
        )
        matrix_df.to_csv(OUTPUT_DIR / f"ap_matrix_{criterion}.csv")

    # Save detailed results table
    rows = []
    for train_ls in LABEL_SETS:
        for eval_ls in LABEL_SETS:
            for ci, criterion in enumerate(CRITERIA):
                ap = results[criterion]["ap"][LABEL_SETS.index(train_ls), LABEL_SETS.index(eval_ls)]
                lo = results[criterion]["lo"][LABEL_SETS.index(train_ls), LABEL_SETS.index(eval_ls)]
                hi = results[criterion]["hi"][LABEL_SETS.index(train_ls), LABEL_SETS.index(eval_ls)]
                sec = secondary.get((train_ls, eval_ls), {}).get(criterion, {})
                rows.append({
                    "trained_on": LABEL_SET_NAMES[train_ls],
                    "evaluated_on": LABEL_SET_NAMES[eval_ls],
                    "criterion": criterion,
                    "AP": ap,
                    "AP_CI_lo": lo,
                    "AP_CI_hi": hi,
                    "AUROC": sec.get("auroc", np.nan),
                    "Precision": sec.get("precision", np.nan),
                    "Recall": sec.get("recall", np.nan),
                    "F1": sec.get("f1", np.nan),
                    "Threshold": sec.get("threshold", np.nan),
                })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "detailed_results.csv", index=False)
    print("  Saved detailed_results.csv")

    # Save disagreement data
    if disagree_info and "disagree_df" in disagree_info:
        disagree_info["disagree_df"].to_csv(OUTPUT_DIR / "c2_disagreement_frames.csv", index=False)
        print("  Saved c2_disagreement_frames.csv")


def print_summary(results, secondary, disagree_info):
    """Print the final interpretive summary."""
    print("\n" + "=" * 70)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 70)

    mv_idx = LABEL_SETS.index("mv")

    # Q1: Does any single-annotator model beat MV model on MV labels?
    print("\n1. Does any annotator-specific model outperform the majority-vote model?")
    for ci, criterion in enumerate(CRITERIA):
        mv_on_mv = results[criterion]["ap"][mv_idx, mv_idx]
        print(f"\n   {criterion} (MV model on MV labels: {mv_on_mv*100:.1f}%):")
        for ri, ls in enumerate(LABEL_SETS):
            if ls == "mv":
                continue
            ann_on_mv = results[criterion]["ap"][ri, mv_idx]
            diff = (ann_on_mv - mv_on_mv) * 100
            marker = ">>>" if diff > 0 else "   "
            print(f"   {marker} {LABEL_SET_NAMES[ls]} on MV: {ann_on_mv*100:.1f}% ({diff:+.1f}%)")

    # Q2: Which annotator produces the most learnable labels?
    print("\n2. Which annotator produces the most learnable/consistent labels?")
    print("   (Diagonal of AP matrix = self-consistency)")
    for ci, criterion in enumerate(CRITERIA):
        print(f"\n   {criterion}:")
        best_ls, best_ap = None, 0
        for ri, ls in enumerate(LABEL_SETS):
            self_ap = results[criterion]["ap"][ri, ri]
            if self_ap > best_ap:
                best_ap = self_ap
                best_ls = ls
            print(f"     {LABEL_SET_NAMES[ls]}: {self_ap*100:.1f}%")
        print(f"     -> Most learnable: {LABEL_SET_NAMES[best_ls]} ({best_ap*100:.1f}%)")

    # Q3: Is label noise a significant performance ceiling?
    print("\n3. Is label noise a significant performance ceiling?")
    for ci, criterion in enumerate(CRITERIA):
        mv_ap = results[criterion]["ap"][mv_idx, mv_idx]
        best_ann_ap = max(
            results[criterion]["ap"][ri, mv_idx]
            for ri in range(3)  # ann1, ann2, ann3
        )
        best_ann = LABEL_SETS[np.argmax([
            results[criterion]["ap"][ri, mv_idx] for ri in range(3)
        ])]
        diff = (best_ann_ap - mv_ap) * 100
        print(f"   {criterion}: Best annotator-trained ({LABEL_SET_NAMES[best_ann]}) on MV: "
              f"{best_ann_ap*100:.1f}% vs MV-trained: {mv_ap*100:.1f}% (diff: {diff:+.1f}%)")

    # Q4: Disagreement analysis
    print("\n4. What does the disagreement analysis reveal?")
    if disagree_info and "disagree_df" in disagree_info:
        ddf = disagree_info["disagree_df"]
        consensus_yes = ddf["model_consensus"].sum()
        total = len(ddf)
        print(f"   C2 disagreement frames: {total}")
        print(f"   Model consensus predicts positive: {consensus_yes}/{total} ({consensus_yes/total*100:.1f}%)")

        # Pattern analysis
        pattern = ddf[
            (ddf["ann1_label"] == 0) & (ddf["ann2_label"] == 1) & (ddf["ann3_label"] == 1)
        ]
        if len(pattern) > 0:
            avg_pred = pattern["avg_pred"].mean()
            print(f"   Ann1=NO, Ann2+Ann3=YES ({len(pattern)} frames): avg model pred = {avg_pred:.3f}")
            if avg_pred >= 0.5:
                print("   -> Models lean towards Ann2+Ann3 (YES) on these frames")
            else:
                print("   -> Models lean towards Ann1 (NO) on these frames")

    # Q5: Implications
    print("\n5. Implications for future work:")
    # Find overall best approach (average across criteria on MV labels)
    mv_maps = []
    for ls in LABEL_SETS:
        ri = LABEL_SETS.index(ls)
        mean_ap = np.mean([results[c]["ap"][ri, mv_idx] for c in CRITERIA])
        mv_maps.append((ls, mean_ap))
    mv_maps.sort(key=lambda x: x[1], reverse=True)
    best_overall = mv_maps[0]
    print(f"   Best overall on MV labels: {LABEL_SET_NAMES[best_overall[0]]} "
          f"(mean AP: {best_overall[1]*100:.1f}%)")
    if best_overall[0] != "mv":
        print(f"   -> Training on {LABEL_SET_NAMES[best_overall[0]]}'s labels is recommended "
              f"over majority vote")
    else:
        print("   -> Majority vote remains the best training label set")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Annotator-Specific Training Experiment")
    print(f"Started: {datetime.datetime.now()}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    df = load_and_prepare_data()

    # Pre-flight checks
    run_preflight_checks(df)

    # Train 4 models
    models = {}
    all_histories = {}
    for idx, label_set in enumerate(LABEL_SETS, 1):
        model, history = train_one_model(df, label_set, idx)
        models[label_set] = model
        all_histories[label_set] = history

    # Full evaluation
    results, secondary, all_preds, all_labels = full_evaluation(models, df)

    # Disagreement analysis
    disagree_info = disagreement_analysis(df, all_preds, all_labels)

    # Visualisations
    print("\n" + "=" * 70)
    print("GENERATING VISUALISATIONS")
    print("=" * 70)
    plot_heatmaps(results)
    plot_bar_charts(results)
    plot_disagreement_scatter(disagree_info)
    plot_training_curves(all_histories)

    # Save results
    save_results(results, secondary, disagree_info, all_histories)

    # Print summary
    print_summary(results, secondary, disagree_info)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
