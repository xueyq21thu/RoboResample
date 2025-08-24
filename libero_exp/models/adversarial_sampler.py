# adversarial_sampler.py

# --- Third-Party Imports ---
from sympy import im
import torch
import numpy as np
import logging
from typing import Dict, Tuple

# --- Local Imports ---
# These are placeholder imports. You should replace them with the actual paths
# to your model definitions.
from .bc_diffusion_policy import BCDPPolicy
from calql.model import Critic, MetaPolicy
from calql.visualize import visualize_critic_evaluation


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
                critic_list: torch.nn.ModuleList = None,
                meta_policy: MetaPolicy = None,
                intervention_k: int = 10,
                meta_policy_buffer: 'MetaPolicyReplayBuffer' = None,
                num_samples: int = 100,
                score_temperature: float = 1.0,
                q_value_threshold: float = 0.0
                ):
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
        self.critic_list = critic_list
        self.meta_policy = meta_policy
        
        self.intervention_threshold = 0.1
        self.device = device
        self.num_samples = num_samples
        self.intervention_k = intervention_k
        self.meta_policy_buffer = meta_policy_buffer
        self.score_temperature = score_temperature
        self.q_value_threshold = q_value_threshold

        # Internal state for tracking HRL progress within an episode
        self.intervention_steps_remaining = 0
        self.decision_point_data = None
        self.sub_trajectory_buffer = []


        # Ensure models are in evaluation mode to disable gradients and dropout
        self.diffusion_policy.eval()
        self.critic.eval()
        # self.critic_list.eval()
        # self.meta_policy.eval()

        logging.info("AdversarialActionSampler initialized.")
        logging.info(f"  - Num Samples per Step: {self.num_samples}")

    def reset(self):
        """Resets the internal HRL state at the beginning of each episode."""
        self.intervention_steps_remaining = 0
        self.decision_point_data = None
        self.sub_trajectory_buffer.clear()
        self.diffusion_policy.reset()

    @torch.no_grad()
    def decide_intervention_timestep(self, data: Dict[str, torch.Tensor]) -> bool:
        """
        (FUNCTION 1) Decides IF an intervention should start at the current timestep.

        This is the high-level decision maker. It uses the MetaPolicy to sample
        a high-level action (exploit vs. explore).

        Returns:
            bool: True if an intervention should start, False otherwise.
        """
        # If we are already in an intervention, we don't need a new decision.
        if self.intervention_steps_remaining > 0:
            return True # Continue the ongoing intervention

        # The sub-trajectory from the previous macro-action is complete.
        # If a decision was made, calculate its reward and store it.
        if self.decision_point_data is not None:
            reward = self._calculate_uncertainty_reward()
            if self.meta_policy_buffer is not None:
                self.meta_policy_buffer.push(
                    log_prob=self.decision_point_data['log_prob'],
                    reward=reward
                )
            self.decision_point_data = None
        
        # Clear the buffer for the next macro-action
        self.sub_trajectory_buffer.clear()

        # --- Ask the MetaPolicy for a new decision ---
        state_vector = torch.cat([data['obs']['gripper_states'], data['obs']['joint_states']], dim=-1).squeeze(1)
        action_dist = self.meta_policy(state_vector)
        high_level_action = action_dist.sample() # 0 for exploit, 1 for explore
        
        if high_level_action.item() == 1: # EXPLORE
            logging.info(f"MetaPolicy decided to EXPLORE for {self.intervention_k} steps.")
            self.intervention_steps_remaining = self.intervention_k
            
            # Store data needed for the delayed reward calculation
            log_prob = action_dist.log_prob(high_level_action)
            self.decision_point_data = {"log_prob": log_prob}
            return True
        else: # EXPLOIT
            return False

    def track_step(self, data: Dict[str, torch.Tensor], executed_action: np.ndarray):
        """Tracks the state and executed action for the uncertainty reward calculation."""
        if self.intervention_steps_remaining > 0 or self.decision_point_data is not None:
            state_vec = torch.cat([data['obs']['gripper_states'], data['obs']['joint_states']], dim=-1).squeeze(1)
            action_tensor = torch.from_numpy(executed_action).to(self.device)
            self.sub_trajectory_buffer.append({"state": state_vec, "action": action_tensor})

    @torch.no_grad()
    def _calculate_uncertainty_reward(self) -> float:
        """Calculates the MetaPolicy's reward based on critic disagreement."""
        if not self.sub_trajectory_buffer:
            return 0.0

        states = torch.stack([item['state'] for item in self.sub_trajectory_buffer])
        actions = torch.stack([item['action'] for item in self.sub_trajectory_buffer])
        
        all_q_values = []
        for critic in self.critic_list:
            q1, _ = critic(states, actions)
            all_q_values.append(q1)
        
        q_values_tensor = torch.stack(all_q_values).squeeze(-1).permute(1, 0)
        uncertainty = torch.std(q_values_tensor, dim=1)
        reward = uncertainty.mean().item()
        logging.info(f"Meta-Policy reward calculated: {reward:.4f}")
        return reward


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
        # log_probs = log_probs.squeeze(0) # Shape: (num_samples,)

        # # --- Step 3: Compute the Combined Adversarial Score ---
        # # We want to MINIMIZE this score.
        # # The score is low for actions with a low Q-value and a high log_prob.
        # # We use -log_prob because we want to minimize that term as well.
        # # The temperature controls the trade-off.
        # adversarial_score = q_values - (self.score_temperature * log_probs)
        
        # # --- Step 4: Select the Action with the Lowest Score ---
        # best_adversarial_index = torch.argmin(adversarial_score)
        # adversarial_action = candidate_actions[0, best_adversarial_index, :].cpu().numpy()

        # logging.info(
        #     f"Adversarial action selected! "
        #     f"Index: {best_adversarial_index}, "
        #     f"Score: {adversarial_score[best_adversarial_index]:.4f} "
        #     f"(Q={q_values[best_adversarial_index]:.4f}, logP={log_probs[best_adversarial_index]:.4f})"
        # )

        # return original_best_action, adversarial_action

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

            # # visualize for sampled actions
            # print("Visualizing critic evaluation for a single sample...")
            # visualize_critic_evaluation(
            #     actions=actions_to_evaluate, 
            #     q_values=q_values,
            #     log_probs=log_probs.squeeze(0),
            #     action_dim_pair1=(0, 1), 
            #     action_dim_pair2=(3, 4)
            # )

            return adversarial_action, True
        else:
            # --- Step 5: Fallback to the Original Best Action ---
            # If no actions were "bad" enough, we don't intervene and let the
            # policy execute its intended best action for stable progress.
            # print("No adversarial action taken.")
            return original_best_action, False

        # selected_action = None
        # # --- Step 4: Select a Random Action ---
        # # if bad_action_indices.numel() > 0:
        # random_index = torch.randint(0, self.num_samples, (1,)).item()
        # selected_action = actions_to_evaluate[random_index].cpu().numpy()
        
        # selected_q_value = q_values[random_index].item()
        # logging.info(
        #     f"Random action selected! "
        #     f"Index: {random_index}, Q-value: {selected_q_value:.4f}."
        # )

        # # --- Step 4: Select the lowest Q-Value Action ---
        # if bad_action_indices.numel() > 0:
        #     lowest_q_index = torch.argmin(q_values).item()
        #     selected_action = actions_to_evaluate[lowest_q_index].cpu().numpy()

        #     selected_q_value = q_values[lowest_q_index].item()
        #     logging.info(
        #         f"Lowest Q-value action selected! "
        #         f"Index: {lowest_q_index}, Q-value: {selected_q_value:.4f}."
        #     )

        # --- Step 5: Insert Fallback Logic ---
        # Fallback in case something went wrong, though it shouldn't in this logic
        # if selected_action is None:
        #      return original_best_action, False

        # return selected_action, True

