import h5py
import numpy as np

# Use one of the files from your log
# IMPORTANT: Make sure the file lock is released first!
HDF5_FILE_PATH = 'rollout/rollout_data_libero_goal.hdf5'

try:
    with h5py.File(HDF5_FILE_PATH, 'r') as f:
        print(f"--- Inspecting file: {HDF5_FILE_PATH} ---")
        
        if 'data' not in f:
            print("ERROR: File does not contain a 'data' group.")
        else:
            data_grp = f['data']
            print(data_grp.keys())
            print(data_grp['demo_0'].keys())
            if 'terminals' not in data_grp:
                print("ERROR: The 'data' group does not have a 'terminals' dataset.")
            else:
                terminals_data = data_grp['terminals'][:]
                print(f"Shape of 'terminals' dataset: {terminals_data.shape}")
                
                num_true_terminals = np.count_nonzero(terminals_data)
                print(f"Number of 'True' terminals found: {num_true_terminals}")

                if len(terminals_data) > 0:
                    print(f"Is the very last terminal flag True? {bool(terminals_data[-1])}")

except Exception as e:
    print(f"An error occurred: {e}")