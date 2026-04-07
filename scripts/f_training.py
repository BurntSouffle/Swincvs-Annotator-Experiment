import torch
import os
from pathlib import Path

def save_weights(model, config, experiment_name):
    """
    Function saving weights of passed model.
    """
    weights_path = Path()
    if config.TRAIN.CHECKPOINT_PATH == None:
        weights_path = Path.cwd() / 'weights'
    else:
        weights_path = config.TRAIN.CHECKPOINT_PATH
    
    print(f"weights path: {weights_path}")

    checkpoint_dir = os.path.join(weights_path, f'{experiment_name}_bestMAP.pt')
    torch.save(model.state_dict(), checkpoint_dir)