# train.py

# --- Standard Library Imports ---
import os
import argparse
import logging
import pickle
from typing import Dict, Any, Tuple, List

# --- Third-Party Imports ---
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

# --- Local Imports ---
from dataset import HDF5CalQLDataset
from core import CalQLLearnerWithHRL

# ==============================================================================
# 1. Configuration Setup
# ==============================================================================

def get_train_config():
    """Parses command-line arguments to configure the HRL training run."""
    parser = argparse.ArgumentParser(description="Train HRL with Cal-QL from offline data.")
    
    # --- Path Arguments ---
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the main HDF5 dataset for the low-level policy.")
    parser.add_argument('--meta_buffer_path', type=str, required=True,
                        help="Path to the collected replay buffer for the high-level meta-policy (a .pkl file).")
    parser.add_argument('--output_dir', type=str, default="./checkpoints/hrl_calql_final",
                        help="Directory to save all model checkpoints.")
    parser.add_argument('--resume_from', type=str, default=None,
                        help="Path to a checkpoint directory to resume training from.")

    # --- Data & Model Dimensions ---
    parser.add_argument('--obs_keys', type=str, nargs='+',
                        default=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
                        help='List of low-dimensional observation keys to concatenate into the state vector.')
    parser.add_argument('--action_dim', type=int, default=7, help='Dimension of the action space.')

    # --- Low-Level Training Hyperparameters ---
    parser.add_argument('--training_steps', type=int, default=500000, help="Total number of gradient steps.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for low-level training.")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate for Actor and Critic optimizers.")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument('--tau', type=float, default=0.005, help="Soft update coefficient for target networks.")
    
    # --- Cal-QL / SAC Hyperparameters ---
    parser.add_argument('--cql_alpha', type=float, default=5.0, help="Weight of the CQL conservative loss term.")
    parser.add_argument('--cql_n_actions', type=int, default=10, help="Number of OOD actions for CQL loss.")
    parser.add_argument('--target_entropy', type=float, default=-7.0, help="Target entropy for SAC's temperature tuning.")
    
    # --- HRL (Meta-Policy) Hyperparameters ---
    parser.add_argument('--num_critics', type=int, default=4,
                        help="Number of critics in the ensemble for uncertainty estimation.")
    parser.add_argument('--meta_policy_lr', type=float, default=1e-4,
                        help="Learning rate for the meta-policy.")
    parser.add_argument('--meta_buffer_capacity', type=int, default=100000,
                        help="Capacity of the meta-policy replay buffer (used in rollout).")
    parser.add_argument('--meta_policy_batch_size', type=int, default=64,
                        help="Batch size for training the meta-policy.")
    parser.add_argument('--meta_policy_train_interval', type=int, default=100,
                        help="How often (in low-level steps) to train the meta-policy.")

    # --- Logging and Saving ---
    parser.add_argument('--log_interval', type=int, default=1000, help="How often to print training logs.")
    parser.add_argument('--save_interval', type=int, default=20000, help="How often to save a model checkpoint.")
    
    return parser.parse_args()

# ==============================================================================
# 2. Main Training Function
# ==============================================================================

def train():
    """Main function to configure and run the entire HRL training process."""
    config = get_train_config()
    
    # --- Setup Logging ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler()])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Low-Level Data ---
    low_level_dataset = HDF5CalQLDataset(
        dataset_path=config.dataset_path, 
        obs_keys=config.obs_keys, 
        gamma=config.gamma
    )
    low_level_dataloader = DataLoader(
        low_level_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    if not low_level_dataset.transitions:
        logging.error(f"No transitions were loaded from {config.dataset_path}.")
        return
    logging.info(f"Successfully loaded {len(low_level_dataset.transitions)} low-level transition steps.")

    # --- Initialize Learner ---
    state_dim = low_level_dataset.transitions[0]['state'].shape[0]
    logging.info(f"Auto-detected state_dim: {state_dim}, using configured action_dim: {config.action_dim}")
    learner = CalQLLearnerWithHRL(state_dim, config.action_dim, config, device)

    # --- Load High-Level Data ---
    try:
        logging.info(f"Loading meta-policy buffer from: {config.meta_buffer_path}")
        with open(config.meta_buffer_path, 'rb') as f:
            learner.meta_policy_buffer.buffer = pickle.load(f)
        logging.info(f"Loaded {len(learner.meta_policy_buffer)} transitions into the meta-policy buffer.")
    except FileNotFoundError:
        logging.warning(f"Meta-policy buffer not found at {config.meta_buffer_path}. Starting with an empty buffer.")
    except Exception as e:
        logging.error(f"Error loading meta-policy buffer: {e}")

    # --- Resume from Checkpoint if specified ---
    start_step = 1
    if config.resume_from:
        learner.load_checkpoint(config.resume_from)
        try:
            start_step = int(os.path.basename(os.path.normpath(config.resume_from)).split('_')[-1]) + 1
            logging.info(f"Training will resume from step {start_step}.")
        except (ValueError, IndexError):
            logging.warning("Could not determine start step from checkpoint path name.")
            
    # --- Training Loop ---
    data_iterator = iter(low_level_dataloader)
    pbar = tqdm.tqdm(
        range(start_step, config.training_steps + 1), 
        desc="Training HRL System", 
        initial=start_step - 1, 
        total=config.training_steps
    )

    for step in pbar:
        try:
            low_level_batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(low_level_dataloader)
            low_level_batch = next(data_iterator)
            
        # --- Train Low-Level Models ---
        low_level_losses = learner.train_low_level_step(low_level_batch)
        
        # --- Train High-Level Model Periodically ---
        if step % config.meta_policy_train_interval == 0:
            meta_losses = learner.train_meta_policy_step()
            # Combine losses for logging
            combined_losses = {**low_level_losses, **meta_losses}
        else:
            combined_losses = low_level_losses

        if step % config.log_interval == 0:
            pbar.set_postfix(combined_losses)
            
        if step % config.save_interval == 0 and step > 0:
            save_path = os.path.join(config.output_dir, f"step_{step}")
            learner.save_checkpoint(save_path)
            logging.info(f"Saved checkpoint to {save_path}")

    # --- Final Save ---
    final_save_path = os.path.join(config.output_dir, "final_model")
    learner.save_checkpoint(final_save_path)
    logging.info(f"Training finished. Final model saved to {final_save_path}")

if __name__ == "__main__":
    train()