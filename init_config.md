# Config Used by RoboReSample

```yaml
  save_rollouts: false
env:
  env_type: libero
  benchmark_envs: ${data.benchmark_envs}
  env_name: ${data.env_name}
  task_id:
  - 2
  env_num: 1
  num_env_rollouts: 50
  horizon: 300
  max_steps: 600
  render_gpu_ids: ${train.train_gpus}
sampler:
  enable: true                 # Master switch to turn adversarial sampling on/off
  
  # Path to the pre-trained Cal-QL critic model
  critic_checkpoint_path: "checkpoints/cal_ql/final_model/critic.pth"
  
  # Dimensions needed to initialize the critic model
  state_dim: 9
  
  # Sampler settings
  num_samples: 50             # Number of actions to sample from the diffusion policy
  q_value_threshold: 0.0      # Q-value below which an action is considered "bad"
  
  # Condition for intervention
  intervention_threshold: 0.01  # Intervene if gripper qpos abs is less than this
```