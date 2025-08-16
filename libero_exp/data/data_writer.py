# data/data_writer.py

import os
import h5py
import numpy as np
import logging

class HDF5Writer:
    """
    A simple utility class to write collected rollout data into an HDF5 file
    that is compatible with the SequenceDataset class.
    """
    def __init__(self, data_path, obs_keys):
        """
        Args:
            data_path (str): Path to the HDF5 file to be created.
            obs_keys (tuple or list): A list of observation keys to be saved.
                                      (e.g., ['robot0_eef_pos', 'agentview_image'])
        """
        self.data_path = data_path
        self.obs_keys = obs_keys
        self._hdf5_file = None
        self._demo_count = 0

        self.obs_keys = [
            "agentview_image", 
            "robot0_eye_in_hand_image", 
            "robot0_gripper_qpos",
            "robot0_joint_pos",
        ]

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        # Create or overwrite the HDF5 file
        self._hdf5_file = h5py.File(self.data_path, 'w')
        self._data_grp = self._hdf5_file.create_group("data")

        logging.info(f"HDF5Writer initialized. Saving data to: {self.data_path}")

    def write_episode(self, episode_data: dict):
        """
        Writes a single complete episode to the HDF5 file.

        Args:
            episode_data (dict): A dictionary containing the episode data.
                                 Expected keys are 'obs', 'next_obs', 'actions',
                                 'rewards', 'dones'.
                                 'obs' and 'next_obs' should be dicts of numpy arrays.
        """
        demo_key = f"demo_{self._demo_count}"
        ep_data_grp = self._data_grp.create_group(demo_key)

        # Write attributes
        num_samples = len(episode_data['actions'])
        ep_data_grp.attrs["num_samples"] = num_samples

        # Write obs
        obs_grp = ep_data_grp.create_group("obs")
        for key in self.obs_keys:
            obs_grp.create_dataset(key, data=np.array(episode_data['obs'][key]), compression='gzip')

        # Write next_obs
        next_obs_grp = ep_data_grp.create_group("next_obs")
        for key in self.obs_keys:
            next_obs_grp.create_dataset(key, data=np.array(episode_data['next_obs'][key]), compression='gzip')
            
        # Write other keys
        ep_data_grp.create_dataset("actions", data=np.array(episode_data['actions']), compression='gzip')
        ep_data_grp.create_dataset("rewards", data=np.array(episode_data['rewards']), compression='gzip')
        ep_data_grp.create_dataset("dones", data=np.array(episode_data['dones']), compression='gzip')
        
        ep_data_grp.create_dataset("terminals", data=np.array(episode_data['dones']), compression='gzip')

        self._demo_count += 1
        logging.info(f"Successfully wrote {demo_key} with {num_samples} steps.")
        logging.info(f"Total demos written so far: {self._demo_count}, demo path: {self.data_path}")

    def close(self):
        """Closes the HDF5 file."""
        if self._hdf5_file:
            self._hdf5_file.close()
            logging.info(f"HDF5Writer closed. Total demos written: {self._demo_count}")