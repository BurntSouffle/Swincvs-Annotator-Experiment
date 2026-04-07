"""
Validate a saved checkpoint on the validation set.
Usage: python validate_checkpoint.py --weights weights/SwinCVS_E2E_MC_IMNP_sd5_bestMAP.pt
"""

import sys
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.f_dataset import get_datasets, get_dataloaders
from scripts.f_build import build_model
from scripts.f_metrics import get_map, get_balanced_accuracies
from scripts.f_environment import get_config

def validate(model, dataloader, device):
    """Run validation and return metrics."""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(tqdm(dataloader, desc="Validating")):
            samples = samples.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs_swin, outputs_lstm = model(samples)

            # Use LSTM outputs for prediction
            preds = torch.sigmoid(outputs_lstm)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate metrics - pass lists directly (get_map concatenates internally)
    C1_ap, C2_ap, C3_ap, mAP = get_map(all_targets, all_preds)

    # Threshold predictions for balanced accuracy
    all_preds_binary = [(p > 0.5).astype(int) for p in all_preds]
    C1_bacc, C2_bacc, C3_bacc, _ = get_balanced_accuracies(all_targets, all_preds_binary)

    return {
        'mAP': mAP,
        'C1_AP': C1_ap,
        'C2_AP': C2_ap,
        'C3_AP': C3_ap,
        'C1_bacc': C1_bacc,
        'C2_bacc': C2_bacc,
        'C3_bacc': C3_bacc,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='config/SwinCVS_config.yaml', help='Config file')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Which split to validate on')
    args = parser.parse_args()

    print("=" * 60)
    print("CHECKPOINT VALIDATION")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print()

    # Load config
    config, experiment_name = get_config(args.config)

    # Resolve dataset directory: env var > config value > cwd
    dataset_dir = (os.environ.get("DATASET_DIR")
                   or config.DATASET_DIR
                   or str(Path.cwd()))
    config.defrost()
    config.DATASET_DIR = dataset_dir
    config.freeze()

    print(f"Experiment: {experiment_name}")
    print(f"Dataset dir: {config.DATASET_DIR}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load datasets
    print("\nLoading datasets...")
    train_ds, val_ds, test_ds = get_datasets(config)
    _, val_loader, test_loader = get_dataloaders(config, train_ds, val_ds, test_ds)

    dataloader = val_loader if args.split == 'val' else test_loader
    print(f"Samples in {args.split} set: {len(dataloader.dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    model = model.to(device)

    # Load weights
    print(f"\nLoading weights from: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        if 'epoch' in checkpoint:
            print(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'mAP' in checkpoint:
            print(f"Saved mAP: {checkpoint['mAP']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    print("Weights loaded successfully!")

    # Run validation
    print(f"\nRunning validation on {args.split} set...")
    metrics = validate(model, dataloader, device)

    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"mAP:     {metrics['mAP']*100:.2f}%")
    print(f"C1 AP:   {metrics['C1_AP']*100:.2f}%")
    print(f"C2 AP:   {metrics['C2_AP']*100:.2f}%")
    print(f"C3 AP:   {metrics['C3_AP']*100:.2f}%")
    print()
    print(f"C1 Balanced Acc: {metrics['C1_bacc']*100:.2f}%")
    print(f"C2 Balanced Acc: {metrics['C2_bacc']*100:.2f}%")
    print(f"C3 Balanced Acc: {metrics['C3_bacc']*100:.2f}%")
    print("=" * 60)

    return metrics

if __name__ == '__main__':
    main()
