# adversarial_sampler.py

# --- Third-Party Imports ---
import torch
import numpy as np
import logging
from typing import Dict, Tuple

# --- Local Imports ---
# These are placeholder imports. You should replace them with the actual paths
# to your model definitions.
from .bc_diffusion_policy import BCDPPolicy
from calql.model import Critic

class AdversarialActionSampler:
    """
    Orchestrates adversarial action sampling for targeted data collection.

    This sampler leverages a generative policy to propose a diverse set of
    candidate actions and a conservative critic to evaluate their quality
    (Q-value). It then strategically selects an action that the critic deems
    "bad" (low Q-value) but was considered plausible by the generative policy.
    This process forces the policy to explore its own failure modes, generating
    valuable data for improving robustness.
    """

    def __init__(self,
                 diffusion_policy: BCDPPolicy,
                 calql_critic: Critic,
                 device: torch.device,
                 num_samples: int = 100,
                 q_value_threshold: float = -1.0):
        """
        Initializes the AdversarialActionSampler.

        Args:
            diffusion_policy (BCDPPolicy): The trained generative diffusion policy.
                It must have a `get_actions(data, num_samples)` method that can
                return a batch of action candidates.
            calql_critic (Critic): The trained Cal-QL critic network used for
                evaluating the quality of actions.
            device (torch.device): The PyTorch device (e.g., 'cuda' or 'cpu')
                on which to perform computations.
            num_samples (int): The number of candidate actions to generate from
                the diffusion policy for each state. A higher number increases
                the chance of finding a suitable adversarial action.
            q_value_threshold (float): A Q-value below which an action is
                considered a "bad" candidate for adversarial selection. This
                should be tuned based on the learned Q-function's scale.
        """
        self.diffusion_policy = diffusion_policy
        self.critic = calql_critic
        self.device = device
        self.num_samples = num_samples
        self.q_value_threshold = q_value_threshold
        self.intervention_threshold = None

        # Ensure models are in evaluation mode to disable gradients and dropout
        self.diffusion_policy.eval()
        self.critic.eval()

        logging.info("AdversarialActionSampler initialized.")
        logging.info(f"  - Num Samples per Step: {self.num_samples}")
        logging.info(f"  - Q-Value Threshold: {self.q_value_threshold}")

    @torch.no_grad()
    def select_action(self, data: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, bool]:
        """
        Generates, evaluates, and selects an action for the given observation.

        The selection process is as follows:
        1.  Generate N candidate actions and their approximate log probabilities
            using the diffusion policy's `get_sampled_actions` method.
        2.  Evaluate the Q-value for all N samples in parallel using the Cal-QL critic.
        3.  Filter for "bad" actions, i.e., those with a Q-value below the threshold.
        4.  If any "bad" actions are found, select the one with the **highest log
            probability**. This is the action the policy was "most confident" was
            good, but the critic identified as bad - a perfect adversarial sample.
        5.  If no actions meet the criteria, gracefully fall back to returning
            the policy's original "best" action (the first sample) to ensure
            stable behavior.

        Args:
            data (Dict[str, torch.Tensor]): Input data dictionary for the policy,
                containing observations and task embeddings, already on the
                correct torch device.

        Returns:
            np.ndarray: The selected action as a NumPy array for the environment.
        """
        # --- Step 1: Generate N Action Samples and Log Probs from Diffusion Policy ---
        # Call the newly named method and request log probabilities.
        candidate_actions, log_probs = self.diffusion_policy.get_sampled_actions(
            data, num_samples=self.num_samples, compute_log_prob=True
        )  # Shapes: (1, num_samples, action_dim), (1, num_samples)

        # The policy's default "best" action is assumed to be the first sample.
        original_best_action = candidate_actions[0, 0, :].cpu().numpy()

        # --- Step 2: Evaluate Q-values for all Candidate Actions ---
        # Prepare state and action tensors for the critic's batch processing.
        gripper_states = data['obs']['gripper_states'] # Shape: (B, T, gripper_dim)
        joint_states = data['obs']['joint_states'] # Shape: (B, T, joint_dim)
        # Repeated for large scale sampling
        state_vector = torch.cat([gripper_states, joint_states], dim=1)

        repeated_state = state_vector.repeat(self.num_samples, 1) # Shape: (num_samples, state_dim)
        actions_to_evaluate = candidate_actions.squeeze(0) # Shape: (num_samples, action_dim)

        # Get Q-values from the critic's Q1 network.
        q1_values, _ = self.critic(repeated_state, actions_to_evaluate)  # Shape: (num_samples, 1)
        q_values = q1_values.squeeze(-1)  # Shape: (num_samples,)

        # --- Step 3: Filter for "Bad" Actions ---
        # Find the indices of actions whose Q-values are below our conservative threshold.
        bad_action_indices = torch.where(q_values < self.q_value_threshold)[0]

        # --- Step 4: Select the Best Adversarial Action based on Log Probability ---
        if bad_action_indices.numel() > 0:
            # If we found at least one "bad" action, we proceed.

            # Get the log probabilities of only the identified "bad" actions.
            log_probs_of_bad_actions = log_probs[0, bad_action_indices]

            # Find the index WITHIN the `bad_action_indices` tensor that corresponds
            # to the highest log probability.
            best_internal_idx = torch.argmax(log_probs_of_bad_actions)
            
            # Map this internal index back to the original index in the full list of candidates.
            best_adversarial_action_index = bad_action_indices[best_internal_idx]

            # Select the final adversarial action.
            adversarial_action = candidate_actions[0, best_adversarial_action_index, :].cpu().numpy()

            logging.info(
                f"Adversarial action inserted! "
                f"Selected action at original index {best_adversarial_action_index} "
                f"with Q-value: {q_values[best_adversarial_action_index]:.4f} and "
                f"log_prob: {log_probs_of_bad_actions[best_internal_idx]:.4f}."
            )

            return adversarial_action, True
        else:
            # --- Step 5: Fallback to the Original Best Action ---
            # If no actions were "bad" enough, we don't intervene and let the
            # policy execute its intended best action for stable progress.
            # print("No adversarial action taken.")
            return original_best_action, False
