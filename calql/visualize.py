# ==============================================================================
# visualize_critic_final.py
#
# A polished, well-documented, and readable script to visualize the Q-value
# landscape of a trained Critic network from Cal-QL.
#
# Features:
#   - Loads a trained Critic model from a specified checkpoint.
#   - Intelligently finds a representative state from an HDF5 dataset based on
#     a specified condition (e.g., gripper closing).
#   - Generates and displays a heatmap of the Q-values over a 2D slice of
#     the action space.
#   - All parameters are configurable via a command-line interface.
#
# Author: Reinsno
# Date: 2025-08-13
# ==============================================================================

# --- Standard Library Imports ---
import os
import argparse
import logging
from typing import Tuple

# --- Third-Party Imports ---
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm

# --- Local Imports ---
# Assuming the model definitions are in a file named 'train_calql_from_hdf5_beautified.py'
# Please adjust the import path if your file is named differently.
from model import Critic
from utils import get_vis_config




# ==============================================================================
# SECTION 2: DATA HANDLING
# ==============================================================================

def find_representative_state(hdf5_path: str, obs_keys: list, gripper_threshold: float) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Finds a representative state from the dataset for visualization.

    The logic is to find a successful trajectory and then, within that
    trajectory, find a key moment, such as when the gripper is closing.

    Args:
        hdf5_path (str): Path to the HDF5 dataset.
        obs_keys (list): List of observation keys that form the state.
        gripper_threshold (float): The threshold for the sum of gripper qpos
                                   to be considered a key state.

    Returns:
        Tuple containing:
        - torch.Tensor: The selected state tensor.
        - torch.Tensor: The action taken at that state.
        - dict: Metadata about the selected state for logging/plotting.
    """
    logging.info(f"Searching for a representative state in {hdf5_path}...")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            demo_keys = list(f['data'].keys())
            # Search in a random order to get different results each run
            np.random.shuffle(demo_keys)
            
            for demo_key in tqdm.tqdm(demo_keys, desc="Searching demos"):
                ep_data = f['data'][demo_key]
                dones = ep_data['dones'][:]
                
                # Check if the trajectory is successful and long enough
                if len(dones) > 10 and bool(dones[-1]):
                    # Found a successful trajectory, now find the key state
                    states = np.concatenate([ep_data['obs'][key][:] for key in obs_keys], axis=1)
                    actions = ep_data['actions'][:]
                    
                    # Search backwards for the first moment the gripper is closing
                    for i in reversed(range(len(states))):
                        state_vec = states[i]
                        # Assuming gripper qpos are the last two dimensions of the state
                        gripper_qpos_sum = state_vec[-2] + state_vec[-1]
                        
                        if gripper_qpos_sum < gripper_threshold:
                            # Found our state!
                            state_of_interest = torch.from_numpy(state_vec.astype(np.float32))
                            action_from_data = torch.from_numpy(actions[i].astype(np.float32))
                            
                            metadata = {
                                "demo_key": demo_key,
                                "step": i,
                                "gripper_width": gripper_qpos_sum
                            }
                            logging.info(f"Found representative state in {demo_key} at step {i} "
                                         f"(gripper width: {gripper_qpos_sum:.4f}).")
                            return state_of_interest, action_from_data, metadata
                            
    except FileNotFoundError:
        logging.error(f"HDF5 file not found at: {hdf5_path}")
        raise
    except Exception as e:
        logging.error(f"Failed to load or process HDF5 file: {e}")
        raise
            
    # If no suitable state is found after checking all demos
    raise ValueError("Could not find any successful trajectory with the specified gripper condition.")


# ==============================================================================
# SECTION 3: VISUALIZATION
# ==============================================================================

def visualize_q_landscape(critic: nn.Module, state: torch.Tensor,
                          action_data: torch.Tensor, metadata: dict, config: argparse.Namespace,
                          device: torch.device):
    """
    Generates and saves a heatmap of the Q-value landscape for a given state.

    Args:
        critic (nn.Module): The trained Critic model.
        state (torch.Tensor): The state at which to visualize the landscape.
        action_data (torch.Tensor): The actual action taken in the data for comparison.
        metadata (dict): Metadata about the state for plotting.
        config (argparse.Namespace): The script's configuration.
        device (torch.device): The device to run inference on.
    """
    logging.info("Generating Q-value landscape heatmap...")
    critic.eval()
    state = state.to(device)
    action_data = action_data.to(device)
    
    # Define the grid for the two action dimensions we are visualizing
    dim_x_idx, dim_y_idx = config.vis_dims
    x_coords = np.linspace(-1.0, 1.0, config.resolution)
    y_coords = np.linspace(-1.0, 1.0, config.resolution)
    q_values = np.zeros((config.resolution, config.resolution))

    # Use the data action as a base, and only vary the visualized dimensions
    base_action = action_data.clone()

    with torch.no_grad():
        for i, x in enumerate(tqdm.tqdm(x_coords, desc=f"Scanning Action Dim {dim_x_idx}")):
            for j, y in enumerate(y_coords):
                # Create an action tensor for this grid point
                action_grid = base_action.clone()
                action_grid[dim_x_idx] = x
                action_grid[dim_y_idx] = y
                
                # The critic expects a batch, so we add a batch dimension
                q1_val, _ = critic(state.unsqueeze(0), action_grid.unsqueeze(0))
                q_values[j, i] = q1_val.item() # Use (j, i) for correct orientation in imshow

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create the heatmap
    img = ax.imshow(q_values, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis', aspect='auto')
    fig.colorbar(img, ax=ax, label='Q-value')
    
    # Mark the position of the actual action from the dataset
    real_action_np = action_data.cpu().numpy()
    ax.scatter(
        real_action_np[dim_x_idx], 
        real_action_np[dim_y_idx], 
        c='red', marker='*', s=300, edgecolor='white', linewidth=1.5,
        label=f'Action from Data (Q ≈ {q_values.max():.2f})'
    )
    
    # --- Beautify the plot ---
    ax.set_xlabel(f"Action Dimension {dim_x_idx}", fontsize=14)
    ax.set_ylabel(f"Action Dimension {dim_y_idx}", fontsize=14)
    title_str = (f"Critic Q-Value Landscape\n"
                 f"Checkpoint: .../{'/'.join(config.critic_path.split('/')[-2:])}\n"
                 f"State from: {metadata['demo_key']} at step {metadata['step']} "
                 f"(Gripper Width: {metadata['gripper_width']:.3f})")
    ax.set_title(title_str, fontsize=16)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # --- Save the figure ---
    filename_base = os.path.basename(os.path.dirname(config.critic_path))
    output_filename = f"q_landscape_{filename_base}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logging.info(f"Heatmap saved to: {output_filename}")
    
    plt.show()


# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """Main function to configure and run the visualization."""
    config = get_vis_config()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler()])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 1. Load the trained Critic model ---
    logging.info(f"Loading Critic model from: {config.critic_path}")
    critic_model = Critic(config.state_dim, config.action_dim).to(device)
    try:
        critic_model.load_state_dict(torch.load(config.critic_path, map_location=device))
    except FileNotFoundError:
        logging.error(f"Critic checkpoint not found at '{config.critic_path}'. Please check the path.")
        return
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    # --- 2. Find a representative state from the dataset ---
    try:
        state, action_data, metadata = find_representative_state(
            config.dataset_path, config.obs_keys, config.gripper_threshold
        )
    except (ValueError, FileNotFoundError) as e:
        logging.error(e)
        return
        
    # --- 3. Generate and display the visualization ---
    visualize_q_landscape(critic_model, state, action_data, metadata, config, device)

if __name__ == "__main__":
    main()