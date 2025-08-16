import argparse

def get_train_config():
    parser = argparse.ArgumentParser(description="Train Cal-QL from HDF5 rollout data.")
    
    # --- Path Arguments ---
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the HDF5 rollout dataset file.")
    parser.add_argument('--output_dir', type=str, default="./checkpoint/cal_ql",
                        help="Directory to save model checkpoints.")
    parser.add_argument('--resume_from', type=str, default=None,
                        help="Path to a checkpoint directory to resume training from.")

    # --- Data Configuration ---
    # These keys should match what was saved in the HDF5 file
    parser.add_argument('--obs_keys', type=str, nargs='+', 
                        default=[
                            "agentview_image", 
                            "robot0_eye_in_hand_image", 
                            "robot0_gripper_qpos",
                            "robot0_joint_pos",
                        ],
                        help='List of low-dimensional observation keys to concatenate into the state vector.')
    parser.add_argument('--action_dim', type=int, default=7, help='Dimension of the action space.')

    # --- Training Hyperparameters ---
    parser.add_argument('--training_steps', type=int, default=500000, help="Total number of gradient steps.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for optimizers.")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor.")
    parser.add_argument('--tau', type=float, default=0.005, help="Soft update coefficient.")
    
    # --- Cal-QL/SAC Hyperparameters ---
    parser.add_argument('--cql_alpha', type=float, default=5.0, help="Weight of the CQL loss.")
    parser.add_argument('--cql_n_actions', type=int, default=10, help="Number of OOD actions for CQL.")
    parser.add_argument('--target_entropy', type=float, default=-7.0, help="Target entropy for SAC.")

    # --- Logging and Saving ---
    parser.add_argument('--log_interval', type=int, default=1000, help="Log interval.")
    parser.add_argument('--save_interval', type=int, default=20000, help="Save interval.")
    
    return parser.parse_args()

def get_vis_config():
    """Parses command-line arguments to configure the visualization."""
    parser = argparse.ArgumentParser(description="Visualize the Q-value landscape of a trained Cal-QL Critic.")
    
    # --- Path Arguments ---
    parser.add_argument('--critic_path', type=str, required=True,
                        help="Path to the trained Critic model checkpoint file (e.g., '.../step_100000/critic.pth').")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the HDF5 dataset file to find a representative state.")

    # --- Model and Data Dimensions ---
    parser.add_argument('--state_dim', type=int, default=8, help="Dimension of the state space.")
    parser.add_argument('--action_dim', type=int, default=7, help="Dimension of the action space.")
    parser.add_argument('--obs_keys', type=str, nargs='+',
                        default=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
                        help='List of low-dimensional observation keys that form the state vector.')

    # --- Visualization Parameters ---
    parser.add_argument('--vis_dims', type=int, nargs=2, default=[0, 1],
                        help="The two action dimensions to visualize on the X and Y axes.")
    parser.add_argument('--resolution', type=int, default=50,
                        help="Resolution of the heatmap grid (e.g., 50 means a 50x50 grid).")
    parser.add_argument('--gripper_threshold', type=float, default=0.6,
                        help="Threshold for gripper width (qpos sum) to identify a key state.")

    return parser.parse_args()