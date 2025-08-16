# ==============================================================================
# train_cal_ql.py
#
# A polished, well-documented, and readable script to train Cal-QL on HDF5
# datasets. This script is designed for clarity, maintainability, and ease of
# use, incorporating best practices for Python and PyTorch development.
#
# Features:
#   - Loads complex HDF5 datasets (e.g., from robomimic rollouts).
#   - Dynamically concatenates specified low-dimensional observations into a
#     flat state vector suitable for MLP policies.
#   - Computes Monte-Carlo returns on-the-fly for Cal-QL's calibration.
#   - Encapsulates all training logic within a clean Learner class.
#   - Supports checkpoint resuming to continue interrupted training runs.
#   - All hyperparameters are managed via a command-line interface.
#
# Author: Reinsno
# Date: 2025-08-11
# ==============================================================================

# --- Standard Library Imports ---
import os
import argparse
import logging
from typing import Dict, Any, Tuple, List

# --- Third-Party Imports ---
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm

from dataset import HDF5CalQLDataset
from core import CalQLLearner
from utils import get_train_config


# -------- Main Training Loop --------
def train():
    """Main function to configure and run the training process."""
    config = get_train_config()
    
    # --- Setup Logging ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler()])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    dataset = HDF5CalQLDataset(
        dataset_path=config.dataset_path, 
        obs_keys=config.obs_keys, 
        gamma=config.gamma
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4, # Use multiple workers to speed up data loading
        pin_memory=True # Speeds up CPU-to-GPU data transfer
    )
    
    if not dataset.transitions:
        logging.error(f"No transitions were loaded from {config.dataset_path}.")
        return
    logging.info(f"Successfully loaded {len(dataset.transitions)} total transition steps.")

    # --- Initialize Learner ---
    state_dim = dataset.transitions[0]['state'].shape[0]
    logging.info(f"Auto-detected state_dim: {state_dim}, using configured action_dim: {config.action_dim}")
    learner = CalQLLearner(state_dim, config.action_dim, config, device)

    # --- Resume from Checkpoint if specified ---
    start_step = 1
    if config.resume_from:
        learner.load_checkpoint(config.resume_from)
        try:
            # Infer start step from checkpoint directory name (e.g., "step_40000")
            start_step = int(os.path.basename(os.path.normpath(config.resume_from)).split('_')[-1]) + 1
            logging.info(f"Training will resume from step {start_step}.")
        except (ValueError, IndexError):
            logging.warning("Could not determine start step from checkpoint path. Resuming logic might be affected.")
            
    # --- Training Loop ---
    data_iterator = iter(dataloader)
    pbar = tqdm.tqdm(
        range(start_step, config.training_steps + 1), 
        desc="Training Cal-QL", 
        initial=start_step - 1, 
        total=config.training_steps
    )

    for step in pbar:
        try:
            batch = next(data_iterator)
        except StopIteration:
            # Restart the iterator when the dataset is exhausted
            data_iterator = iter(dataloader)
            batch = next(data_iterator)
            
        losses = learner.train_step(batch)
        
        if step % config.log_interval == 0:
            pbar.set_postfix(losses)
            
        if step % config.save_interval == 0:
            save_path = os.path.join(config.output_dir, f"step_{step}")
            learner.save_checkpoint(save_path)
            logging.info(f"Saved checkpoint to {save_path}")

    # --- Final Save ---
    final_save_path = os.path.join(config.output_dir, "final_model")
    learner.save_checkpoint(final_save_path)
    logging.info(f"Training finished. Final model saved to {final_save_path}")

if __name__ == "__main__":
    train()