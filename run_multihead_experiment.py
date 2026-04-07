"""
SwinCVS Multi-Head + Optimal Labels Experiment
================================================
Phase 0: Verify existing SwinCVS checkpoint as Model A (no retraining)
Phase 1: Train Model B — multi-head SwinCVS, MV labels
Phase 2: Train Model C — multi-head SwinCVS, optimal labels
Phase 3: Comparison table and analysis

Resume-safe: checkpoints saved every epoch; restarts pick up where they left off.
"""

import os
import sys
import time
import json
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Add SwinCVS to path — works from any location
SWINCVS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SWINCVS_DIR))

from scripts.f_environment import get_config, verify_results_weights_folder
from scripts.f_dataset import (
    check_dataset, get_three_dataframes, get_dataloaders,
    EndoscapesSwinCVS_Dataset, get_transform_sequence
)
from scripts.f_build import build_model
from scripts.f_training_utils import build_optimizer, update_params, NativeScalerWithGradNormCount
from scripts.f_metrics import get_map, get_balanced_accuracies
from scripts.m_swinv2 import SwinTransformerV2, load_pretrained
from scripts.m_swincvs import SwinCVSModel
from scripts.m_swincvs_multihead import SwinCVSMultiHeadModel
from sklearn.metrics import average_precision_score


def reset_seed(seed):
    """Reset random seeds without strict deterministic mode (avoids CUDA crashes)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


OUTPUT_DIR = SWINCVS_DIR / "experiment_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_N = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPECTED_A = {"C1": 65.18, "C2": 65.11, "C3": 74.00}
TOLERANCE_PP = 1.0  # percentage points


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_backbone(config):
    """Build SwinV2 backbone with pretrained weights."""
    model = SwinTransformerV2(
        img_size=384,
        patch_size=config.BACKBONE.SWINV2.PATCH_SIZE,
        in_chans=config.BACKBONE.SWINV2.IN_CHANS,
        num_classes=config.BACKBONE.NUM_CLASSES,
        embed_dim=config.BACKBONE.SWINV2.EMBED_DIM,
        depths=config.BACKBONE.SWINV2.DEPTHS,
        num_heads=config.BACKBONE.SWINV2.NUM_HEADS,
        window_size=config.BACKBONE.SWINV2.WINDOW_SIZE,
        mlp_ratio=config.BACKBONE.SWINV2.MLP_RATIO,
        qkv_bias=config.BACKBONE.SWINV2.QKV_BIAS,
        drop_rate=config.BACKBONE.DROP_RATE,
        drop_path_rate=config.BACKBONE.DROP_PATH_RATE,
        ape=config.BACKBONE.SWINV2.APE,
        patch_norm=config.BACKBONE.SWINV2.PATCH_NORM,
        use_checkpoint=config.BACKBONE.USE_CHECKPOINT,
        pretrained_window_sizes=config.BACKBONE.SWINV2.PRETRAINED_WINDOW_SIZES,
    )
    if "swinv2_base_patch4" in config.BACKBONE.PRETRAINED:
        load_pretrained(config, model)
    model.head = nn.Identity()
    return model


def build_multihead_model(config):
    """Build SwinCVS with multi-head classification."""
    backbone = build_backbone(config)
    return SwinCVSMultiHeadModel(backbone, config)


def build_multihead_optimizer(config, model):
    """Build optimizer for multi-head model — same LR structure as original."""
    if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
        params = [
            {"params": model.swinv2_model.parameters(), "lr": config.TRAIN.OPTIMIZER.ENCODER_LR},
            {"params": model.fc_swin_c1.parameters(), "lr": config.TRAIN.OPTIMIZER.ENCODER_LR},
            {"params": model.fc_swin_c2.parameters(), "lr": config.TRAIN.OPTIMIZER.ENCODER_LR},
            {"params": model.fc_swin_c3.parameters(), "lr": config.TRAIN.OPTIMIZER.ENCODER_LR},
            {"params": model.lstm.parameters(), "lr": config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
            {"params": model.fc_c1.parameters(), "lr": config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
            {"params": model.fc_c2.parameters(), "lr": config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
            {"params": model.fc_c3.parameters(), "lr": config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
        ]
    else:
        params = [
            {"params": model.swinv2_model.parameters(), "lr": config.TRAIN.OPTIMIZER.ENCODER_LR},
            {"params": model.lstm.parameters(), "lr": config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
            {"params": model.fc_c1.parameters(), "lr": config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
            {"params": model.fc_c2.parameters(), "lr": config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
            {"params": model.fc_c3.parameters(), "lr": config.TRAIN.OPTIMIZER.CLASSIFIER_LR},
        ]
    optimizer = torch.optim.AdamW(
        params, eps=config.TRAIN.OPTIMIZER.EPS,
        betas=config.TRAIN.OPTIMIZER.BETAS,
        weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY,
    )
    return optimizer


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING WITH CUSTOM JSON
# ═══════════════════════════════════════════════════════════════════════════════

def get_datasets_custom_json(config, json_name="annotation_ds_coco.json"):
    """Load datasets using a specific JSON file name for labels."""
    dataset_dir = Path(config.DATASET_DIR) / "endoscapes"
    assert dataset_dir.exists(), f"Dataset not found at {dataset_dir}"
    print(f"  Dataset loaded from: {dataset_dir}")
    print(f"  Using label JSON: {json_name}")

    import scripts.f_dataset as ds_module
    original_get_three = ds_module.get_three_dataframes

    def patched_get_three(image_folder, lstm=False):
        train_dir = image_folder / "train"
        val_dir = image_folder / "val"
        test_dir = image_folder / "test"

        train_dataframe = ds_module.get_dataframe(train_dir / json_name)
        val_dataframe = ds_module.get_dataframe(val_dir / json_name)
        test_dataframe = ds_module.get_dataframe(test_dir / json_name)

        if lstm:
            with open(image_folder / "all" / "annotation_coco.json", "r") as f:
                all_images = json.load(f)
            all_image_names = [x["file_name"] for x in all_images["images"]]
            train_images = [img for img in all_image_names if 1 <= int(img.split("_")[0]) <= 120]
            val_images = [img for img in all_image_names if 121 <= int(img.split("_")[0]) <= 161]
            test_images = [img for img in all_image_names if 162 <= int(img.split("_")[0]) <= 201]

            train_dataframe = ds_module.add_unlabelled_imgs(train_images, train_dataframe)
            val_dataframe = ds_module.add_unlabelled_imgs(val_images, val_dataframe)
            test_dataframe = ds_module.add_unlabelled_imgs(test_images, test_dataframe)

            train_dataframe = ds_module.get_frame_sequence_dataframe(train_dataframe, train_dir)
            val_dataframe = ds_module.get_frame_sequence_dataframe(val_dataframe, val_dir)
            test_dataframe = ds_module.get_frame_sequence_dataframe(test_dataframe, test_dir)
            return train_dataframe, val_dataframe, test_dataframe

        updated_train = ds_module.update_dataframe(train_dataframe, train_dir)
        updated_val = ds_module.update_dataframe(val_dataframe, val_dir)
        updated_test = ds_module.update_dataframe(test_dataframe, test_dir)
        return updated_train, updated_val, updated_test

    ds_module.get_three_dataframes = patched_get_three
    try:
        train_df, val_df, test_df = patched_get_three(dataset_dir, lstm=config.MODEL.LSTM)
    finally:
        ds_module.get_three_dataframes = original_get_three

    transform = get_transform_sequence(config)
    train_dataset = EndoscapesSwinCVS_Dataset(train_df, transform)
    val_dataset = EndoscapesSwinCVS_Dataset(val_df, transform)
    test_dataset = EndoscapesSwinCVS_Dataset(test_df, transform)

    return train_dataset, val_dataset, test_dataset


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def collect_predictions(model, loader):
    """Run model on a dataloader, return (probs_np, targets_np)."""
    model.eval()
    all_probs, all_targets = [], []
    with torch.inference_mode():
        for samples, targets in loader:
            samples, targets = samples.to(DEVICE), targets.to(DEVICE)
            output = model(samples)
            if isinstance(output, tuple):
                output = output[1]
            prob = torch.sigmoid(output)
            all_probs.append(prob.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


def bootstrap_ap(probs, targets):
    """Compute per-criterion AP with bootstrap CIs."""
    results = {}
    for ci, cname in enumerate(["C1", "C2", "C3"]):
        y_true = targets[:, ci]
        y_score = probs[:, ci]
        ap = average_precision_score(y_true, y_score)

        rng = np.random.RandomState(42)
        boot = []
        for _ in range(BOOTSTRAP_N):
            idx = rng.randint(0, len(y_true), len(y_true))
            bt, bs = y_true[idx], y_score[idx]
            if bt.sum() == 0 or bt.sum() == len(bt):
                continue
            boot.append(average_precision_score(bt, bs))
        lo = np.percentile(boot, 2.5) if len(boot) > 10 else np.nan
        hi = np.percentile(boot, 97.5) if len(boot) > 10 else np.nan
        results[cname] = {"ap": ap, "lo": lo, "hi": hi}

    mean_ap = np.mean([results[c]["ap"] for c in ["C1", "C2", "C3"]])
    results["mean"] = {"ap": mean_ap}
    return results


def fmt_result(results, criterion):
    """Format a single criterion result as 'XX.XX% [lo-hi]'."""
    r = results[criterion]
    if "lo" in r and not np.isnan(r.get("lo", np.nan)):
        return f"{r['ap']*100:.2f}% [{r['lo']*100:.1f}-{r['hi']*100:.1f}]"
    return f"{r['ap']*100:.2f}%"


def print_results(results, label=""):
    """Print C1/C2/C3/Mean AP with CIs."""
    if label:
        print(f"  {label}:")
    for c in ["C1", "C2", "C3"]:
        print(f"    {c}: {fmt_result(results, c)}")
    print(f"    Mean: {results['mean']['ap']*100:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING WITH RESUME SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, optimizer, config, train_loader, val_loader, model_name):
    """Train a SwinCVS model with checkpoint resume support.

    Saves checkpoint_latest.pt after every epoch and best_model.pt on improvement.
    Returns (model with best weights loaded, training history dict).
    """
    save_dir = OUTPUT_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / "checkpoint_latest.pt"
    best_path = save_dir / "best_model.pt"

    loss_scaler = NativeScalerWithGradNormCount()
    class_weights = torch.tensor(config.TRAIN.CLASS_WEIGHTS).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(weight=class_weights).to(DEVICE)

    num_epochs = config.TRAIN.EPOCHS
    best_mAP = 0.0
    best_state = None
    best_epoch = 0
    start_epoch = 0

    multiclassifier_alpha = config.TRAIN.MULTICLASSIFIER_ALPHA if config.MODEL.MULTICLASSIFIER else None
    multiclassifier_beta = 1 - multiclassifier_alpha if multiclassifier_alpha else None

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_map": [], "val_c1_ap": [], "val_c2_ap": [], "val_c3_ap": []}

    # ── Resume from checkpoint if it exists ──
    if ckpt_path.exists():
        print(f"  Found checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        loss_scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]  # epoch is 1-indexed: saved after completing epoch N
        best_mAP = ckpt["best_val_map"]
        best_epoch = ckpt.get("best_epoch", start_epoch)
        history = ckpt["history"]
        # Recompute alpha/beta to match the resumed epoch
        if multiclassifier_alpha is not None:
            alpha = config.TRAIN.MULTICLASSIFIER_ALPHA
            beta = 1 - alpha
            for e in range(start_epoch):
                alpha, beta = update_params(alpha, beta, e)
            multiclassifier_alpha = alpha
            multiclassifier_beta = beta
        if best_mAP > 0 and best_path.exists():
            best_state = torch.load(best_path, weights_only=True)
        print(f"  Resuming from epoch {start_epoch + 1} (best mAP so far: {best_mAP:.4f} @ epoch {best_epoch})")
    else:
        print(f"  Starting fresh training")

    if start_epoch >= num_epochs:
        print(f"  Training already complete ({num_epochs} epochs). Skipping.")
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name} — epochs {start_epoch+1}..{num_epochs}")
    print(f"{'='*70}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params:,} total, {trainable:,} trainable")

    epoch_times = []

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Update multiclassifier weights (no-op for epochs <= 4)
        if config.MODEL.E2E and config.MODEL.MULTICLASSIFIER:
            multiclassifier_alpha, multiclassifier_beta = update_params(
                multiclassifier_alpha, multiclassifier_beta, epoch
            )

        # ── Training ──
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
                loss = multiclassifier_alpha * criterion(outputs_swin, targets) + \
                       multiclassifier_beta * criterion(outputs_lstm, targets)
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

        # ── Validation ──
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

                prob = torch.sigmoid(outputs_lstm)
                val_probs.append(prob.cpu())
                val_targets.append(targets.cpu())

                loss = criterion(outputs_lstm, targets)
                val_loss += loss.item()
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

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        remaining = np.mean(epoch_times) * (num_epochs - epoch - 1)
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))

        print(f"  [{model_name}] Epoch {epoch+1}/{num_epochs} | "
              f"Train: {train_loss:.1f} | Val: {val_loss:.1f} | "
              f"mAP: {mAP:.4f} (C1:{C1_ap:.3f} C2:{C2_ap:.3f} C3:{C3_ap:.3f})"
              f"{improved} | {epoch_elapsed:.0f}s | ETA {remaining_str}")

        # ── Save checkpoint ──
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": loss_scaler.state_dict(),
            "epoch": epoch + 1,
            "best_val_map": best_mAP,
            "best_epoch": best_epoch,
            "history": history,
        }, ckpt_path)

    # ── Save training history CSV ──
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    print(f"  Best val mAP: {best_mAP:.4f} at epoch {best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: VERIFY BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

def phase0_verify_baseline(config, test_loader):
    """Load existing SwinCVS checkpoint, evaluate, and return Model A results.

    Stops if any criterion deviates from expected by more than TOLERANCE_PP.
    """
    print("\n" + "=" * 70)
    print("PHASE 0: VERIFY EXISTING CHECKPOINT AS MODEL A")
    print("=" * 70)

    results_path = OUTPUT_DIR / "Model_A" / "test_results.json"

    # Check if Phase 0 already completed in a prior run
    if results_path.exists():
        print(f"  Found saved Model A results: {results_path}")
        with open(results_path) as f:
            results = json.load(f)
        print_results(results, "Model A (loaded from cache)")
        print("Phase 0 PASSED — baseline verified (cached)")
        return results

    ckpt_path = SWINCVS_DIR / "weights" / "SwinCVS_E2E_MC_IMNP_sd5_bestMAP.pt"
    print(f"  Loading: {ckpt_path}")

    model = build_model(config)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(DEVICE)
    model.inference = True  # disable multiclassifier dual output

    probs, targets = collect_predictions(model, test_loader)
    results = bootstrap_ap(probs, targets)

    print_results(results, "Model A test results")

    # ── Verify against expected values ──
    print(f"\n  Verification (tolerance: {TOLERANCE_PP}pp):")
    failed = False
    for c in ["C1", "C2", "C3"]:
        actual = results[c]["ap"] * 100
        expected = EXPECTED_A[c]
        diff = abs(actual - expected)
        status = "OK" if diff <= TOLERANCE_PP else "FAIL"
        print(f"    {c}: {actual:.2f}% (expected {expected:.2f}%, diff {diff:.2f}pp) [{status}]")
        if diff > TOLERANCE_PP:
            failed = True

    if failed:
        print("\n  STOPPING: Baseline verification failed. Fix before proceeding.")
        sys.exit(1)

    # ── Save results ──
    save_dir = OUTPUT_DIR / "Model_A"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    del model
    torch.cuda.empty_cache()

    print("\nPhase 0 PASSED — baseline verified")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: TRAIN MODEL B
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_train_model_b(config, seed, mv_train_loader, mv_val_loader, mv_test_loader):
    """Train multi-head SwinCVS with MV labels."""
    print("\n" + "=" * 70)
    print("PHASE 1: MODEL B — Multi-Head + MV Labels")
    print("=" * 70)

    results_path = OUTPUT_DIR / "Model_B" / "test_results.json"

    # Check if already fully trained and evaluated
    if results_path.exists():
        print(f"  Found saved Model B results: {results_path}")
        with open(results_path) as f:
            results = json.load(f)
        history_path = OUTPUT_DIR / "Model_B" / "training_history.csv"
        history = pd.read_csv(history_path).to_dict(orient="list") if history_path.exists() else None
        print_results(results, "Model B (loaded from cache)")
        return results, history

    reset_seed(seed)
    model = build_multihead_model(config).to(DEVICE)
    optimizer = build_multihead_optimizer(config, model)
    model, history = train_model(model, optimizer, config, mv_train_loader, mv_val_loader, "Model_B")

    print("  Evaluating Model B on test set (MV labels)...")
    probs, targets = collect_predictions(model, mv_test_loader)
    results = bootstrap_ap(probs, targets)
    print_results(results, "Model B test results")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    del model, optimizer
    torch.cuda.empty_cache()
    return results, history


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: TRAIN MODEL C
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_train_model_c(config, seed, opt_train_dataset, opt_train_loader, opt_val_loader, mv_test_loader):
    """Train multi-head SwinCVS with optimal labels, evaluate on MV test."""
    print("\n" + "=" * 70)
    print("PHASE 2: MODEL C — Multi-Head + Optimal Labels")
    print("=" * 70)

    results_path = OUTPUT_DIR / "Model_C" / "test_results.json"

    # Check if already fully trained and evaluated
    if results_path.exists():
        print(f"  Found saved Model C results: {results_path}")
        with open(results_path) as f:
            results = json.load(f)
        history_path = OUTPUT_DIR / "Model_C" / "training_history.csv"
        history = pd.read_csv(history_path).to_dict(orient="list") if history_path.exists() else None
        print_results(results, "Model C (loaded from cache)")
        return results, history

    # Recompute class weights for optimal labels
    config_c = copy.deepcopy(config)
    opt_labels = np.array([opt_train_dataset[i][1].numpy() for i in range(len(opt_train_dataset))])
    c1_pos = opt_labels[:, 0].mean()
    c2_pos = opt_labels[:, 1].mean()
    c3_pos = opt_labels[:, 2].mean()
    new_weights = [(1 - c1_pos) / max(c1_pos, 0.01),
                   (1 - c2_pos) / max(c2_pos, 0.01),
                   (1 - c3_pos) / max(c3_pos, 0.01)]
    config_c.TRAIN.CLASS_WEIGHTS = new_weights
    print(f"  Optimal label class weights: C1={new_weights[0]:.2f}, C2={new_weights[1]:.2f}, C3={new_weights[2]:.2f}")
    print(f"  Original MV class weights:   {list(config.TRAIN.CLASS_WEIGHTS)}")

    reset_seed(seed)
    model = build_multihead_model(config_c).to(DEVICE)
    optimizer = build_multihead_optimizer(config_c, model)
    model, history = train_model(model, optimizer, config_c, opt_train_loader, opt_val_loader, "Model_C")

    # Evaluate on MV test labels (standard benchmark)
    print("  Evaluating Model C on test set (MV labels for comparability)...")
    probs, targets = collect_predictions(model, mv_test_loader)
    results = bootstrap_ap(probs, targets)
    print_results(results, "Model C test results (MV labels)")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    del model, optimizer
    torch.cuda.empty_cache()
    return results, history


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_comparison(results_a, results_b, results_c, hist_b, hist_c):
    """Print comparison table, analysis, and save plots."""
    print("\n" + "=" * 70)
    print("PHASE 3: COMPARISON (all evaluated on MV test labels)")
    print("=" * 70)

    published = {
        "C1": {"ap": 0.6423}, "C2": {"ap": 0.6250},
        "C3": {"ap": 0.6703}, "mean": {"ap": 0.6459}
    }

    all_results = {
        "Published SwinCVS E2E": published,
        "Model A: Original Ckpt": results_a,
        "Model B: Multi-Head+MV": results_b,
        "Model C: Multi-Head+Opt": results_c,
    }

    # ── Table ──
    print(f"\n{'Model':<26} {'C1 AP':>10} {'C2 AP':>10} {'C3 AP':>10} {'Mean AP':>10}")
    print("-" * 70)
    rows = []
    for name, res in all_results.items():
        c1 = res["C1"]["ap"] * 100
        c2 = res["C2"]["ap"] * 100
        c3 = res["C3"]["ap"] * 100
        mean = res["mean"]["ap"] * 100
        print(f"{name:<26} {c1:>8.2f}% {c2:>8.2f}% {c3:>8.2f}% {mean:>8.2f}%")
        rows.append({"Model": name, "C1_AP": c1, "C2_AP": c2, "C3_AP": c3, "Mean_AP": mean})

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)

    # ── Analysis ──
    a_mean = results_a["mean"]["ap"] * 100
    b_mean = results_b["mean"]["ap"] * 100
    c_mean = results_c["mean"]["ap"] * 100

    print(f"\n  1. Head separation alone (B vs A): {b_mean:.2f}% vs {a_mean:.2f}% ({b_mean - a_mean:+.2f}pp)")
    print(f"     {'HELPS' if b_mean > a_mean + 0.5 else 'HURTS' if b_mean < a_mean - 0.5 else 'NEUTRAL'}")

    print(f"\n  2. Optimal labels on top (C vs B): {c_mean:.2f}% vs {b_mean:.2f}% ({c_mean - b_mean:+.2f}pp)")
    print(f"     {'HELPS' if c_mean > b_mean + 0.5 else 'HURTS' if c_mean < b_mean - 0.5 else 'NEUTRAL'}")

    c_c2 = results_c["C2"]["ap"] * 100
    b_c2 = results_b["C2"]["ap"] * 100
    print(f"\n  3. C2 collapse check: Model C C2={c_c2:.2f}%, Model B C2={b_c2:.2f}% ({c_c2 - b_c2:+.2f}pp)")
    if c_c2 < b_c2 - 5:
        print("     WARNING: C2 collapse detected (same pattern as ViT mixed model)")
    else:
        print("     No C2 collapse")

    best_name = max(["Model A: Original Ckpt", "Model B: Multi-Head+MV", "Model C: Multi-Head+Opt"],
                    key=lambda n: all_results[n]["mean"]["ap"])
    print(f"\n  4. Best overall: {best_name} ({all_results[best_name]['mean']['ap']*100:.2f}%)")

    # ── Training curves ──
    if hist_b is not None and hist_c is not None:
        plot_training_curves(hist_b, hist_c)
    plot_comparison_bar(all_results)


def plot_training_curves(hist_b, hist_c):
    """Plot per-epoch training curves for Model B and C overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Model B": "#e74c3c", "Model C": "#2ecc71"}

    for idx, metric in enumerate(["val_map", "train_loss", "val_loss"]):
        ax = axes[idx]
        for name, hist in [("Model B", hist_b), ("Model C", hist_c)]:
            ax.plot(hist["epoch"], hist[metric], label=name,
                    color=colors[name], linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Epoch")
        titles = {"val_map": "Val mAP", "train_loss": "Train Loss", "val_loss": "Val Loss"}
        ax.set_title(titles[metric], fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("SwinCVS Training Curves: Model B vs C", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved training_curves.png")


def plot_comparison_bar(all_results):
    """Bar chart comparing all models."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    colors = {"Published SwinCVS E2E": "#7f8c8d", "Model A: Original Ckpt": "#3498db",
              "Model B: Multi-Head+MV": "#e74c3c", "Model C: Multi-Head+Opt": "#2ecc71"}

    models = list(all_results.keys())
    for idx, metric in enumerate(["C1", "C2", "C3", "mean"]):
        ax = axes[idx]
        vals = [all_results[m][metric]["ap"] * 100 for m in models]
        x = np.arange(len(models))
        ax.bar(x, vals, color=[colors.get(m, "#bdc3c7") for m in models],
               edgecolor="black", linewidth=0.5)

        for i, m in enumerate(models):
            r = all_results[m][metric]
            if "lo" in r and not np.isnan(r.get("lo", np.nan)):
                ax.errorbar(i, vals[i], yerr=[[vals[i] - r["lo"] * 100], [r["hi"] * 100 - vals[i]]],
                            fmt="none", ecolor="black", capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels([m.split(":")[0] if ":" in m else m[:12] for m in models],
                           rotation=35, ha="right", fontsize=8)
        title = f"{metric} AP (%)" if metric != "mean" else "Mean AP (%)"
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 85)
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("SwinCVS Experiment Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved comparison_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    print("SwinCVS Multi-Head + Optimal Labels Experiment")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.chdir(SWINCVS_DIR)

    # Load config (allow --config and --dataset_dir overrides)
    import argparse
    parser = argparse.ArgumentParser(description="SwinCVS Multi-Head Experiment")
    parser.add_argument("--config", type=str, default=str(SWINCVS_DIR / "config" / "SwinCVS_config.yaml"))
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Root dir containing 'endoscapes/' folder")
    args = parser.parse_args()

    config_path = args.config
    config, experiment_name = get_config(config_path)

    # Resolve dataset directory: CLI arg > env var > config value > parent of SwinCVS dir
    dataset_dir = (args.dataset_dir
                   or os.environ.get("DATASET_DIR")
                   or config.DATASET_DIR
                   or str(SWINCVS_DIR.parent))
    config.defrost()
    config.DATASET_DIR = dataset_dir
    config.freeze()
    print(f"Dataset dir: {config.DATASET_DIR}")

    seed = config.SEED
    reset_seed(seed)

    # ── Load MV datasets (used for A, B, and test evaluation of C) ──
    print("\nLoading MV (original) datasets...")
    mv_train, mv_val, mv_test = get_datasets_custom_json(config, "annotation_ds_coco.json")
    mv_train_loader, mv_val_loader, mv_test_loader = get_dataloaders(config, mv_train, mv_val, mv_test)
    print(f"  Train: {len(mv_train)}, Val: {len(mv_val)}, Test: {len(mv_test)}")

    # ── Load Optimal datasets ──
    print("\nLoading Optimal label datasets...")
    opt_train, opt_val, opt_test = get_datasets_custom_json(config, "annotation_ds_coco_optimal.json")
    opt_train_loader, opt_val_loader, opt_test_loader = get_dataloaders(config, opt_train, opt_val, opt_test)
    print(f"  Train: {len(opt_train)}, Val: {len(opt_val)}, Test: {len(opt_test)}")

    # ── Phase 0 ──
    results_a = phase0_verify_baseline(config, mv_test_loader)

    # ── Phase 1 ──
    results_b, hist_b = phase1_train_model_b(config, seed, mv_train_loader, mv_val_loader, mv_test_loader)

    # ── Phase 2 ──
    results_c, hist_c = phase2_train_model_c(config, seed, opt_train, opt_train_loader, opt_val_loader, mv_test_loader)

    # ── Phase 3 ──
    phase3_comparison(results_a, results_b, results_c, hist_b, hist_c)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE — {elapsed/60:.1f} minutes total")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
