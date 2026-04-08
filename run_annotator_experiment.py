"""
SwinCVS Annotator-Specific Training Experiment
================================================
Tests whether training on consistent single-annotator labels outperforms
majority vote for SwinCVS — the temporal SOTA model with SwinV2 + LSTM.

Trains 3 SwinCVS models (Ann1, Ann2, Ann3) sequentially, then evaluates
all 4 (+ existing MV baseline) against all label sets.

Usage:
    python -u run_annotator_experiment.py \
        --config config/SwinCVS_config_runpod.yaml \
        --dataset_dir /workspace
"""

import os
import sys
import ast
import time
import json
import copy
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from scripts.f_environment import get_config
from scripts.f_dataset import (
    get_dataloaders, EndoscapesSwinCVS_Dataset, get_transform_sequence,
    get_dataframe, add_unlabelled_imgs, get_frame_sequence_dataframe
)
from scripts.f_build import build_model
from scripts.f_training_utils import build_optimizer, update_params, NativeScalerWithGradNormCount
from scripts.f_metrics import get_map
from sklearn.metrics import average_precision_score

# ── Constants ────────────────────────────────────────────────────
TRAIN_MODELS = [
    # (key, display_name, json_filename, output_subdir)
    ("ann1", "Annotator 1", "annotation_ds_coco_ann1.json", "Model_Ann1"),
    ("ann2", "Annotator 2", "annotation_ds_coco_ann2.json", "Model_Ann2"),
    ("ann3", "Annotator 3", "annotation_ds_coco_ann3.json", "Model_Ann3"),
]
ALL_MODEL_KEYS = ["ann1", "ann2", "ann3", "mv"]
EVAL_LABEL_KEYS = ["ann1", "ann2", "ann3", "mv", "and23"]
LABEL_NAMES = {
    "ann1": "Annotator 1", "ann2": "Annotator 2", "ann3": "Annotator 3",
    "mv": "Majority Vote", "and23": "Ann2+3 AND",
}
JSON_NAMES = {
    "ann1": "annotation_ds_coco_ann1.json",
    "ann2": "annotation_ds_coco_ann2.json",
    "ann3": "annotation_ds_coco_ann3.json",
    "mv": "annotation_ds_coco.json",
    "and23": "annotation_ds_coco_and23.json",
}
MODEL_DIRS = {"ann1": "Model_Ann1", "ann2": "Model_Ann2", "ann3": "Model_Ann3"}

BOOTSTRAP_N = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = SCRIPT_DIR / "annotator_outputs"

# Reference values (percentages)
MODEL_A_AP = {"C1": 65.18, "C2": 65.11, "C3": 74.00}
PUBLISHED_E2E = {"C1": 64.23, "C2": 62.50, "C3": 67.03}
VIT_ANN2_MV = {"C1": 49.6, "C2": 40.6, "C3": 54.9}
VIT_MV_MV = {"C1": 44.6, "C2": 41.2, "C3": 53.8}


def reset_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


# ═════════════════════════════════════════════════════════════════
# LABEL JSON GENERATION
# ═════════════════════════════════════════════════════════════════

def parse_votes(vote_str):
    try:
        parsed = ast.literal_eval(vote_str)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
            return [int(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    return None


def generate_label_jsons(endoscapes_dir):
    """Create per-annotator and AND23 label JSONs for each split if missing."""
    metadata_csv = endoscapes_dir / "all_metadata.csv"
    assert metadata_csv.exists(), f"Metadata not found: {metadata_csv}"

    needed = []
    for label_key in ["ann1", "ann2", "ann3", "and23"]:
        for split in ["train", "val", "test"]:
            if not (endoscapes_dir / split / JSON_NAMES[label_key]).exists():
                needed.append(label_key)
                break

    if not needed:
        print("All annotator label JSONs already exist.")
        return

    print(f"Generating label JSONs for: {needed}")
    meta = pd.read_csv(metadata_csv)
    meta = meta[meta["is_ds_keyframe"] == True].copy()

    for ann_idx in [1, 2, 3]:
        col = f"cvs_annotator_{ann_idx}"
        parsed = meta[col].apply(parse_votes)
        for ci in range(3):
            meta[f"C{ci+1}_ann{ann_idx}"] = parsed.apply(lambda x, c=ci: x[c] if x else np.nan)

    lookups = {}
    for ann_idx in [1, 2, 3]:
        key = f"ann{ann_idx}"
        if key not in needed:
            continue
        labels = {}
        for _, row in meta.iterrows():
            fk = f"{int(row['vid'])}_{int(row['frame'])}"
            c1 = row[f"C1_ann{ann_idx}"]
            if np.isnan(c1):
                continue
            labels[fk] = [float(row[f"C1_ann{ann_idx}"]),
                          float(row[f"C2_ann{ann_idx}"]),
                          float(row[f"C3_ann{ann_idx}"])]
        lookups[key] = labels

    if "and23" in needed:
        and23 = {}
        for _, row in meta.iterrows():
            fk = f"{int(row['vid'])}_{int(row['frame'])}"
            if np.isnan(row.get("C1_ann2", np.nan)) or np.isnan(row.get("C1_ann3", np.nan)):
                continue
            and23[fk] = [
                1.0 if (row["C1_ann2"] == 1 and row["C1_ann3"] == 1) else 0.0,
                1.0 if (row["C2_ann2"] == 1 and row["C2_ann3"] == 1) else 0.0,
                1.0 if (row["C3_ann2"] == 1 and row["C3_ann3"] == 1) else 0.0,
            ]
        lookups["and23"] = and23

    for label_key, lookup in lookups.items():
        json_name = JSON_NAMES[label_key]
        for split in ["train", "val", "test"]:
            out_path = endoscapes_dir / split / json_name
            if out_path.exists():
                continue

            with open(endoscapes_dir / split / "annotation_ds_coco.json") as f:
                data = json.load(f)

            modified = 0
            for img in data["images"]:
                fname = img["file_name"].split(".")[0]
                if fname in lookup:
                    img["ds"] = lookup[fname]
                    modified += 1

            with open(out_path, "w") as f:
                json.dump(data, f)

            n = len(data["images"])
            rates = []
            for ci in range(3):
                pos = sum(1 for img in data["images"] if round(img["ds"][ci]) == 1)
                rates.append(f"C{ci+1}+={pos/n*100:.1f}%")
            print(f"  {label_key}/{split}: {modified}/{n} | {' '.join(rates)}")

    print()


# ═════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═════════════════════════════════════════════════════════════════

def load_swincvs_datasets(config, json_name):
    """Load SwinCVS (5-frame LSTM) datasets with a specific label JSON."""
    dataset_dir = Path(config.DATASET_DIR) / "endoscapes"
    import scripts.f_dataset as ds_mod

    dfs = {}
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        df = ds_mod.get_dataframe(split_dir / json_name)

        if config.MODEL.LSTM:
            with open(dataset_dir / "all" / "annotation_coco.json") as f:
                all_imgs = json.load(f)
            all_names = [x["file_name"] for x in all_imgs["images"]]
            vid_ranges = {"train": (1, 120), "val": (121, 161), "test": (162, 201)}
            lo, hi = vid_ranges[split]
            split_imgs = [img for img in all_names if lo <= int(img.split("_")[0]) <= hi]
            df = ds_mod.add_unlabelled_imgs(split_imgs, df)
            df = ds_mod.get_frame_sequence_dataframe(df, split_dir)

        dfs[split] = df

    transform = get_transform_sequence(config)
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = EndoscapesSwinCVS_Dataset(dfs[split], transform)
        ds._label_dataframe = dfs[split]
        datasets[split] = ds

    return datasets["train"], datasets["val"], datasets["test"]


def compute_class_weights(dataset):
    """Compute BCEWithLogitsLoss pos weights from training labels."""
    labels = np.array(dataset._label_dataframe["classification"].tolist())
    weights = []
    for ci in range(3):
        pos = labels[:, ci].mean()
        weights.append(float((1 - pos) / max(pos, 0.01)))
    return weights


# ═════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════

def train_annotator_model(config, seed, train_loader, val_loader, model_name):
    """Train original SwinCVS (single-head) with checkpoint resume.

    Returns (model_with_best_weights, history_dict).
    """
    save_dir = OUTPUT_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / "checkpoint_latest.pt"
    best_path = save_dir / "best_model.pt"

    reset_seed(seed)
    model = build_model(config).to(DEVICE)
    optimizer = build_optimizer(config, model)

    loss_scaler = NativeScalerWithGradNormCount()
    class_weights = torch.tensor(config.TRAIN.CLASS_WEIGHTS).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(weight=class_weights).to(DEVICE)

    num_epochs = config.TRAIN.EPOCHS
    best_mAP = 0.0
    best_state = None
    best_epoch = 0
    start_epoch = 0

    mc_alpha = config.TRAIN.MULTICLASSIFIER_ALPHA if config.MODEL.MULTICLASSIFIER else None
    mc_beta = (1 - mc_alpha) if mc_alpha is not None else None

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_map": [], "val_c1_ap": [], "val_c2_ap": [], "val_c3_ap": []}

    # ── Resume ──
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        loss_scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_mAP = ckpt["best_val_map"]
        best_epoch = ckpt.get("best_epoch", start_epoch)
        history = ckpt["history"]
        if mc_alpha is not None:
            a, b = config.TRAIN.MULTICLASSIFIER_ALPHA, 1 - config.TRAIN.MULTICLASSIFIER_ALPHA
            for e in range(start_epoch):
                a, b = update_params(a, b, e)
            mc_alpha, mc_beta = a, b
        if best_mAP > 0 and best_path.exists():
            best_state = torch.load(best_path, weights_only=True)
        print(f"  Resuming from epoch {start_epoch + 1} "
              f"(best mAP: {best_mAP:.4f} @ epoch {best_epoch})")

    if start_epoch >= num_epochs:
        print(f"  Already complete ({num_epochs} epochs).")
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params:,} total, {trainable:,} trainable")
    print(f"  Class weights: [{', '.join(f'{w:.2f}' for w in config.TRAIN.CLASS_WEIGHTS)}]")

    epoch_times = []

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
            mc_alpha, mc_beta = update_params(mc_alpha, mc_beta, epoch)

        # ── Train ──
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        num_batches = len(train_loader)

        for idx, (samples, targets) in enumerate(train_loader):
            samples, targets = samples.to(DEVICE), targets.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=True):
                if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                    outputs_swin, outputs_lstm = model(samples)
                else:
                    outputs_lstm = model(samples)

            if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                loss = mc_alpha * criterion(outputs_swin, targets) + \
                       mc_beta * criterion(outputs_lstm, targets)
            else:
                loss = criterion(outputs_lstm, targets)

            is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                        parameters=model.parameters(), create_graph=is_second_order,
                        update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
            optimizer.zero_grad()
            train_loss += loss.item()
            torch.cuda.synchronize()

            if (idx + 1) % 200 == 0 or (idx + 1) == num_batches:
                print(f"    batch {idx+1}/{num_batches}", end="\r")

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_probs, val_targets = [], []

        with torch.inference_mode():
            for samples, targets in val_loader:
                samples, targets = samples.to(DEVICE), targets.to(DEVICE)
                if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
                    _, outputs_lstm = model(samples)
                else:
                    outputs_lstm = model(samples)

                val_probs.append(torch.sigmoid(outputs_lstm).cpu())
                val_targets.append(targets.cpu())
                val_loss += criterion(outputs_lstm, targets).item()
                torch.cuda.synchronize()

        C1_ap, C2_ap, C3_ap, mAP = get_map(val_targets, val_probs)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(mAP)
        history["val_c1_ap"].append(C1_ap)
        history["val_c2_ap"].append(C2_ap)
        history["val_c3_ap"].append(C3_ap)

        improved = ""
        if mAP >= best_mAP:
            best_mAP = mAP
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, best_path)
            improved = " *BEST*"

        elapsed = time.time() - epoch_start
        epoch_times.append(elapsed)
        eta = np.mean(epoch_times) * (num_epochs - epoch - 1)

        print(f"  [{model_name}] Epoch {epoch+1}/{num_epochs} | "
              f"Train: {train_loss:.1f} | Val: {val_loss:.1f} | "
              f"mAP: {mAP:.4f} (C1:{C1_ap:.3f} C2:{C2_ap:.3f} C3:{C3_ap:.3f})"
              f"{improved} | {elapsed:.0f}s | ETA {time.strftime('%H:%M:%S', time.gmtime(eta))}")

        # ── Checkpoint ──
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": loss_scaler.state_dict(),
            "epoch": epoch + 1,
            "best_val_map": best_mAP,
            "best_epoch": best_epoch,
            "history": history,
        }, ckpt_path)

    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    print(f"  Best val mAP: {best_mAP:.4f} at epoch {best_epoch}\n")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ═════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═════════════════════════════════════════════════════════════════

def collect_predictions(model, loader):
    """Run inference, return (probs_np, targets_np)."""
    model.eval()
    all_probs, all_targets = [], []
    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.to(DEVICE)
            output = model(samples)
            if isinstance(output, tuple):
                output = output[1]  # LSTM output
            all_probs.append(torch.sigmoid(output).cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


def bootstrap_ap(y_true, y_score):
    """AP with bootstrap 95% CI. Returns (ap%, lo%, hi%)."""
    ap = average_precision_score(y_true, y_score) * 100
    rng = np.random.RandomState(42)
    boot = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.randint(0, len(y_true), len(y_true))
        bt, bs = y_true[idx], y_score[idx]
        if bt.sum() == 0 or bt.sum() == len(bt):
            continue
        boot.append(average_precision_score(bt, bs) * 100)
    lo = np.percentile(boot, 2.5) if len(boot) > 10 else np.nan
    hi = np.percentile(boot, 97.5) if len(boot) > 10 else np.nan
    return ap, lo, hi


# ═════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════

def plot_training_curves(histories):
    """Overlaid training curves for all 3 annotator models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Model_Ann1": "#e74c3c", "Model_Ann2": "#2ecc71", "Model_Ann3": "#3498db"}

    for idx, metric in enumerate(["val_map", "train_loss", "val_loss"]):
        ax = axes[idx]
        for model_dir, hist in histories.items():
            if hist is None:
                continue
            label = model_dir.replace("Model_", "").replace("Ann", "Annotator ")
            ax.plot(hist["epoch"], hist[metric], label=label,
                    color=colors.get(model_dir, "#999"), linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Epoch")
        titles = {"val_map": "Val mAP", "train_loss": "Train Loss", "val_loss": "Val Loss"}
        ax.set_title(titles[metric], fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("SwinCVS Annotator Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved training_curves.png")


def plot_ap_heatmaps(ap_matrices):
    """Heatmaps for C1, C2, C3 AP matrices (4 models x 5 label sets)."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))

    for idx, cname in enumerate(["C1", "C2", "C3"]):
        ax = axes[idx]
        matrix = ap_matrices[cname]
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")

        ax.set_xticks(range(len(EVAL_LABEL_KEYS)))
        ax.set_xticklabels([LABEL_NAMES[k] for k in EVAL_LABEL_KEYS],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(ALL_MODEL_KEYS)))
        ax.set_yticklabels([LABEL_NAMES[k] for k in ALL_MODEL_KEYS], fontsize=9)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = "white" if matrix[i, j] > 60 else "black"
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                        fontsize=8, color=color)

        ax.set_title(f"{cname} AP (%)", fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.6, label="AP (%)")
    plt.suptitle("SwinCVS Annotator Experiment: AP Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ap_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved ap_heatmaps.png")


def plot_comparison_bar(results_on_mv):
    """Bar chart comparing all models on MV labels."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    models = list(results_on_mv.keys())
    colors_map = {
        "SwinCVS Ann1": "#e74c3c", "SwinCVS Ann2": "#2ecc71",
        "SwinCVS Ann3": "#3498db", "SwinCVS MV (A)": "#f39c12",
        "Published E2E": "#7f8c8d", "ViT Ann2 (ref)": "#9b59b6",
        "ViT MV (ref)": "#bdc3c7",
    }

    for idx, metric in enumerate(["C1", "C2", "C3", "Mean"]):
        ax = axes[idx]
        vals = [results_on_mv[m][metric] for m in models]
        x = np.arange(len(models))
        ax.bar(x, vals, color=[colors_map.get(m, "#95a5a6") for m in models],
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{metric} AP (%)", fontweight="bold")
        ax.set_ylim(0, 85)
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=7, fontweight="bold")

    plt.suptitle("SwinCVS Annotator Experiment: Evaluated on MV Labels",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved comparison_bar.png")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    parser = argparse.ArgumentParser(description="SwinCVS Annotator Experiment")
    parser.add_argument("--config", type=str,
                        default=str(SCRIPT_DIR / "config" / "SwinCVS_config.yaml"))
    parser.add_argument("--dataset_dir", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("SwinCVS ANNOTATOR-SPECIFIC TRAINING EXPERIMENT")
    print("=" * 70)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.chdir(SCRIPT_DIR)

    config, _ = get_config(args.config)
    dataset_dir = (args.dataset_dir
                   or os.environ.get("DATASET_DIR")
                   or config.DATASET_DIR
                   or str(SCRIPT_DIR.parent))
    config.defrost()
    config.DATASET_DIR = dataset_dir
    config.freeze()
    print(f"Dataset dir: {config.DATASET_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = config.SEED
    endoscapes_dir = Path(config.DATASET_DIR) / "endoscapes"

    # ══════════════════════════════════════════════════════════════
    # STEP 1: GENERATE LABEL JSONs
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: GENERATE ANNOTATOR LABEL JSONs")
    print("=" * 70)
    generate_label_jsons(endoscapes_dir)

    # ══════════════════════════════════════════════════════════════
    # STEP 2: TRAIN 3 MODELS SEQUENTIALLY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 2: SEQUENTIAL TRAINING")
    print("=" * 70)

    histories = {}

    for ann_key, ann_name, json_name, model_dir in TRAIN_MODELS:
        print(f"\n{'=' * 70}")
        print(f"TRAINING: {model_dir} ({ann_name} labels)")
        print(f"{'=' * 70}")

        hist_path = OUTPUT_DIR / model_dir / "training_history.csv"
        if hist_path.exists():
            print(f"  Already trained. Loading history from disk.")
            histories[model_dir] = pd.read_csv(hist_path).to_dict(orient="list")
            continue

        # Load train + val datasets
        print(f"  Loading datasets ({json_name})...")
        train_ds, val_ds, _ = load_swincvs_datasets(config, json_name)
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        # Compute class weights for this annotator's labels
        weights = compute_class_weights(train_ds)
        print(f"  Pos weights: C1={weights[0]:.2f}, C2={weights[1]:.2f}, C3={weights[2]:.2f}")

        config_ann = copy.deepcopy(config)
        config_ann.defrost()
        config_ann.TRAIN.CLASS_WEIGHTS = weights
        config_ann.freeze()

        train_loader, val_loader, _ = get_dataloaders(config_ann, train_ds, val_ds, val_ds)

        model, history = train_annotator_model(config_ann, seed, train_loader, val_loader, model_dir)
        histories[model_dir] = history

        del model, train_ds, val_ds, train_loader, val_loader
        torch.cuda.empty_cache()

    print("\nAll 3 models trained.")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: FULL EVALUATION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: EVALUATION (4 models x 5 label sets)")
    print("=" * 70)

    # Load any test set for inference (images identical across label sets)
    print("\n  Loading test set for inference...")
    _, _, mv_test_ds = load_swincvs_datasets(config, "annotation_ds_coco.json")
    num_workers = int(os.environ.get("NUM_WORKERS", "0"))
    test_loader = DataLoader(mv_test_ds, batch_size=1, shuffle=False,
                             pin_memory=True, num_workers=num_workers)
    n_test = len(mv_test_ds)
    print(f"  Test sequences: {n_test}")

    # ── Collect predictions for each model ──
    print("\n  Running inference...")
    all_probs = {}

    for ann_key, ann_name, _, model_dir in TRAIN_MODELS:
        best_path = OUTPUT_DIR / model_dir / "best_model.pt"
        assert best_path.exists(), f"Missing checkpoint: {best_path}"

        reset_seed(seed)
        model = build_model(config).to(DEVICE)
        model.load_state_dict(torch.load(best_path, weights_only=True))
        probs, _ = collect_predictions(model, test_loader)
        all_probs[ann_key] = probs
        print(f"    {ann_name}: done ({probs.shape[0]} samples)")
        del model; torch.cuda.empty_cache()

    # MV model from existing checkpoint
    mv_ckpt = SCRIPT_DIR / "weights" / "SwinCVS_E2E_MC_IMNP_sd5_bestMAP.pt"
    assert mv_ckpt.exists(), f"Missing MV checkpoint: {mv_ckpt}"
    print(f"  Loading MV model from {mv_ckpt.name}...")
    model = build_model(config).to(DEVICE)
    model.load_state_dict(torch.load(mv_ckpt, weights_only=True))
    probs, _ = collect_predictions(model, test_loader)
    all_probs["mv"] = probs
    print(f"    Majority Vote: done ({probs.shape[0]} samples)")
    del model; torch.cuda.empty_cache()

    # ── Load ground truth for all 5 label sets ──
    print("\n  Loading ground truth labels...")
    all_gt = {}
    for lk in EVAL_LABEL_KEYS:
        _, _, test_ds = load_swincvs_datasets(config, JSON_NAMES[lk])
        gt = np.array(test_ds._label_dataframe["classification"].tolist())
        all_gt[lk] = gt
        assert len(gt) == n_test, \
            f"Label count mismatch for {lk}: got {len(gt)}, expected {n_test}"
        print(f"    {LABEL_NAMES[lk]}: {gt.shape}")

    # ── Compute AP matrices ──
    print("\n  Computing AP matrices with bootstrap CIs...")
    ap_matrices = {}   # cname -> np.array(4, 5)
    ap_details = {}    # cname -> {(model_key, label_key): (ap, lo, hi)}

    for ci, cname in enumerate(["C1", "C2", "C3"]):
        matrix = np.zeros((len(ALL_MODEL_KEYS), len(EVAL_LABEL_KEYS)))
        details = {}
        for mi, mk in enumerate(ALL_MODEL_KEYS):
            for li, lk in enumerate(EVAL_LABEL_KEYS):
                ap, lo, hi = bootstrap_ap(all_gt[lk][:, ci], all_probs[mk][:, ci])
                matrix[mi, li] = ap
                details[(mk, lk)] = (ap, lo, hi)
        ap_matrices[cname] = matrix
        ap_details[cname] = details

    # ── Print AP matrices ──
    print("\n" + "=" * 70)
    print("AP MATRICES (trained on rows, evaluated on columns)")
    print("=" * 70)

    for cname in ["C1", "C2", "C3"]:
        print(f"\n  {cname} AP:")
        header = f"  {'':>16}" + "".join(f"{LABEL_NAMES[lk]:>16}" for lk in EVAL_LABEL_KEYS)
        print(header)
        print("  " + "-" * (16 + 16 * len(EVAL_LABEL_KEYS)))
        for mi, mk in enumerate(ALL_MODEL_KEYS):
            row = f"  {LABEL_NAMES[mk]:>16}"
            for li, lk in enumerate(EVAL_LABEL_KEYS):
                ap, lo, hi = ap_details[cname][(mk, lk)]
                row += f" {ap:5.1f} [{lo:4.1f}-{hi:4.1f}]"
            print(row)

        df = pd.DataFrame(ap_matrices[cname],
                          index=[LABEL_NAMES[mk] for mk in ALL_MODEL_KEYS],
                          columns=[LABEL_NAMES[lk] for lk in EVAL_LABEL_KEYS])
        df.to_csv(OUTPUT_DIR / f"ap_matrix_{cname}.csv")

    # ── Comparison table (MV eval column) ──
    mv_li = EVAL_LABEL_KEYS.index("mv")

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (evaluated on MV labels)")
    print("=" * 70)
    print(f"\n  {'Model':<22} {'C1 AP':>8} {'C2 AP':>8} {'C3 AP':>8} {'Mean AP':>8}")
    print("  " + "-" * 54)

    comparison_rows = []
    results_on_mv = {}

    for mk in ALL_MODEL_KEYS:
        mi = ALL_MODEL_KEYS.index(mk)
        c1 = ap_matrices["C1"][mi, mv_li]
        c2 = ap_matrices["C2"][mi, mv_li]
        c3 = ap_matrices["C3"][mi, mv_li]
        mean = (c1 + c2 + c3) / 3
        name = f"SwinCVS {LABEL_NAMES[mk]}" if mk != "mv" else "SwinCVS MV (A)"
        print(f"  {name:<22} {c1:>7.2f}% {c2:>7.2f}% {c3:>7.2f}% {mean:>7.2f}%")
        comparison_rows.append({"Model": name, "C1": c1, "C2": c2, "C3": c3, "Mean": mean})
        results_on_mv[name] = {"C1": c1, "C2": c2, "C3": c3, "Mean": mean}

    for ref_name, ref in [("Published E2E", PUBLISHED_E2E),
                           ("ViT Ann2 (ref)", VIT_ANN2_MV),
                           ("ViT MV (ref)", VIT_MV_MV)]:
        mean = (ref["C1"] + ref["C2"] + ref["C3"]) / 3
        print(f"  {ref_name:<22} {ref['C1']:>7.2f}% {ref['C2']:>7.2f}% {ref['C3']:>7.2f}% {mean:>7.2f}%")
        comparison_rows.append({"Model": ref_name, **ref, "Mean": mean})
        results_on_mv[ref_name] = {"C1": ref["C1"], "C2": ref["C2"], "C3": ref["C3"], "Mean": mean}

    pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)

    # ── Plots ──
    print("\n  Generating plots...")
    for _, _, _, md in TRAIN_MODELS:
        if md not in histories or histories[md] is None:
            hp = OUTPUT_DIR / md / "training_history.csv"
            if hp.exists():
                histories[md] = pd.read_csv(hp).to_dict(orient="list")
    plot_training_curves(histories)
    plot_ap_heatmaps(ap_matrices)
    plot_comparison_bar(results_on_mv)

    # ══════════════════════════════════════════════════════════════
    # SUMMARY: KEY FINDINGS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY: KEY FINDINGS")
    print("=" * 70)

    mv_mean_baseline = (MODEL_A_AP["C1"] + MODEL_A_AP["C2"] + MODEL_A_AP["C3"]) / 3

    # Q1
    print(f"\n1. Does any annotator-trained SwinCVS beat Model A "
          f"(MV, {mv_mean_baseline:.2f}%) on MV labels?")
    for ak in ["ann1", "ann2", "ann3"]:
        mi = ALL_MODEL_KEYS.index(ak)
        mean = (ap_matrices["C1"][mi, mv_li] + ap_matrices["C2"][mi, mv_li]
                + ap_matrices["C3"][mi, mv_li]) / 3
        diff = mean - mv_mean_baseline
        marker = ">>>" if diff > 0 else "   "
        print(f"   {marker} {LABEL_NAMES[ak]}: {mean:.2f}% ({diff:+.2f}pp)")

    # Q2
    print("\n2. Highest per-criterion AP on MV labels:")
    for ci, cn in enumerate(["C1", "C2", "C3"]):
        best_k, best_v = None, -1
        for mk in ALL_MODEL_KEYS:
            v = ap_matrices[cn][ALL_MODEL_KEYS.index(mk), mv_li]
            if v > best_v:
                best_v, best_k = v, mk
        mv_v = ap_matrices[cn][ALL_MODEL_KEYS.index("mv"), mv_li]
        print(f"   {cn}: {LABEL_NAMES[best_k]} ({best_v:.2f}%, {best_v - mv_v:+.2f}pp vs MV)")

    # Q3
    print("\n3. Does the ViT finding transfer -- is Ann2 still the best single strategy?")
    ann_means = {}
    for ak in ["ann1", "ann2", "ann3"]:
        mi = ALL_MODEL_KEYS.index(ak)
        ann_means[ak] = (ap_matrices["C1"][mi, mv_li] + ap_matrices["C2"][mi, mv_li]
                         + ap_matrices["C3"][mi, mv_li]) / 3
        print(f"   {LABEL_NAMES[ak]} mean on MV: {ann_means[ak]:.2f}%")
    best_ann = max(ann_means, key=ann_means.get)
    if best_ann == "ann2":
        print(f"   -> YES, Ann2 is best ({ann_means[best_ann]:.2f}%)")
    else:
        print(f"   -> NO, {LABEL_NAMES[best_ann]} is best ({ann_means[best_ann]:.2f}%)")

    # Q4
    print("\n4. Does Ann3's lenient C1 labels improve C1 AP? (ViT: +6.9pp)")
    ann3_c1 = ap_matrices["C1"][ALL_MODEL_KEYS.index("ann3"), mv_li]
    mv_c1 = ap_matrices["C1"][ALL_MODEL_KEYS.index("mv"), mv_li]
    diff = ann3_c1 - mv_c1
    print(f"   Ann3 C1={ann3_c1:.2f}% vs MV C1={mv_c1:.2f}% ({diff:+.2f}pp)")
    if diff > 2:
        print("   -> YES, substantial improvement")
    elif diff > 0:
        print("   -> Marginal improvement")
    else:
        print("   -> NO improvement")

    # Q5
    print("\n5. Is C2 still label-insensitive? (ViT range was ~10pp)")
    c2_vals = {}
    for mk in ALL_MODEL_KEYS:
        c2_vals[mk] = ap_matrices["C2"][ALL_MODEL_KEYS.index(mk), mv_li]
        print(f"   {LABEL_NAMES[mk]} C2: {c2_vals[mk]:.2f}%")
    c2_range = max(c2_vals.values()) - min(c2_vals.values())
    print(f"   Range: {c2_range:.2f}pp {'(insensitive)' if c2_range < 5 else '(sensitive -- temporal changes things)'}")

    # Q6
    print("\n6. Self-consistency (diagonal = most learnable):")
    diag_means = {}
    for mk in ALL_MODEL_KEYS:
        mi = ALL_MODEL_KEYS.index(mk)
        li = EVAL_LABEL_KEYS.index(mk)
        vals = [ap_matrices[cn][mi, li] for cn in ["C1", "C2", "C3"]]
        diag_means[mk] = np.mean(vals)
        print(f"   {LABEL_NAMES[mk]}: C1={vals[0]:.1f}% C2={vals[1]:.1f}% C3={vals[2]:.1f}% "
              f"| Mean={diag_means[mk]:.1f}%")
    best_diag = max(diag_means, key=diag_means.get)
    print(f"   -> Most learnable: {LABEL_NAMES[best_diag]} ({diag_means[best_diag]:.1f}%)")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE -- {elapsed / 60:.1f} minutes total")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
