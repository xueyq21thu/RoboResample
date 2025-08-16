import os
import glob
import h5py
import tqdm
import torch
import logging
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple

class HDF5CalQLDataset(Dataset):
    """
    A custom PyTorch Dataset that automatically discovers and loads data from all
    HDF5 files within a specified directory.

    It correctly reads the HDF5 structure where each file contains multiple '/data/demo_x'
    groups, with each group representing a single episode and containing its own
    'actions', 'dones', 'terminals', etc. datasets.
    """
    def __init__(self, dataset_path: str, obs_keys: List[str], gamma: float):
        self.dataset_path = dataset_path
        self.obs_keys = obs_keys
        self.gamma = gamma
        self.transitions = []
        self._load_and_process_data()

    def _discover_hdf5_files(self) -> List[str]:
        """Finds all HDF5 files in the specified directory."""
        if not os.path.isdir(self.dataset_path):
            logging.error(f"Dataset path is not a valid directory: {self.dataset_path}")
            return []
        
        search_pattern = os.path.join(self.dataset_path, "*.hdf5")
        hdf5_files = glob.glob(search_pattern)
        
        if not hdf5_files:
            logging.warning(f"No HDF5 files found in directory: {self.dataset_path}")
        
        return sorted(hdf5_files)

    def _load_and_process_data(self):
        """
        Internal method to read all HDF5 files by iterating through 'demo_x' groups
        and preparing all transitions.
        """
        hdf5_paths = self._discover_hdf5_files()
        
        if not hdf5_paths:
            return

        logging.info(f"Found {len(hdf5_paths)} HDF5 file(s) to process.")
        
        for hdf5_path in hdf5_paths:
            logging.info(f"Processing file: {hdf5_path}")
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    if 'data' not in f:
                        logging.warning(f"File {hdf5_path} does not contain a 'data' group. Skipping.")
                        continue
                        
                    demo_keys = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
                    
                    for demo_key in tqdm.tqdm(demo_keys, desc=f"Processing demos in {os.path.basename(hdf5_path)}"):
                        ep_data = f['data'][demo_key]
                        num_samples = ep_data.attrs['num_samples']
                        if num_samples < 1:
                            continue

                        # Correctly access datasets within the ep_data group
                        actions = ep_data['actions'][:]
                        dones = ep_data['dones'][:]
                        # # The 'terminals' key exists inside each demo group
                        # terminals = ep_data['terminals'][:]

                        state_keys = [key for key in self.obs_keys if 'image' not in key]
                        if state_keys:
                            states = np.concatenate([ep_data['obs'][key][:] for key in state_keys], axis=1)
                            next_states = np.concatenate([ep_data['next_obs'][key][:] for key in state_keys], axis=1)
                        else:
                            states = np.zeros((num_samples, 0), dtype=np.float32)
                            next_states = np.zeros((num_samples, 0), dtype=np.float32)

                        # Compute Monte-Carlo returns for this episode
                        final_success = bool(dones[-1])
                        mc_returns = np.zeros(num_samples, dtype=np.float32)
                        mc_return = 0.0
                        for i in reversed(range(num_samples)):
                            reward = 1.0 if (i == num_samples - 1 and final_success) else 0.0
                            mc_return = reward + self.gamma * mc_return
                            mc_returns[i] = mc_return

                        # Append transitions, ensuring correct shapes
                        for i in range(num_samples):
                            self.transitions.append({
                                "state": states[i].astype(np.float32),
                                "action": actions[i].astype(np.float32),
                                "reward": np.array([1.0 if (i == num_samples - 1 and final_success) else 0.0], dtype=np.float32),
                                "next_state": next_states[i].astype(np.float32),
                                "done": np.array(dones[i], dtype=np.float32),
                                "mc_return": np.array([mc_returns[i]], dtype=np.float32)
                            })
                
            except Exception as e:
                logging.error(f"Failed to load or process HDF5 file {hdf5_path}: {e}")
                continue
        
        logging.info(f"Finished processing all files. Total transitions loaded: {len(self.transitions)}")

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Tuple:
        trans = self.transitions[idx]
        return (
            torch.from_numpy(trans["state"]),
            torch.from_numpy(trans["action"]),
            torch.from_numpy(trans["reward"]),
            torch.from_numpy(trans["next_state"]),
            torch.from_numpy(trans["done"]),
            # Note: Added terminal to the output if you need it later
            # torch.from_numpy(trans["terminal"]), 
            torch.from_numpy(trans["mc_return"])
        )