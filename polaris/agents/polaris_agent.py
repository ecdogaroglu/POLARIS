import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from polaris.algorithms.regularization.si import (
    SILoss,
    calculate_path_integral_from_replay_buffer,
)

from ..agents.memory.replay_buffer import ReplayBuffer
from ..networks.temporal_gnn import TemporalGNN
from ..networks.policy import ContinuousPolicyNetwork, PolicyNetwork
from ..networks.qnetwork import QNetwork
from ..networks.transformer import TransformerBeliefProcessor
from ..utils.device import get_best_device
from ..utils.encoding import encode_observation


class ArchitectureMismatchError(ValueError):
    """Raised when there's a mismatch between saved and current model architectures."""
    pass


class ModelComponentError(RuntimeError):
    """Raised when model components cannot be loaded due to architecture changes."""
    pass


class POLARISAgent:
    """POLARIS agent for social learning with GNN-based inference."""

    def __init__(
        self,
        agent_id,
        num_agents,
        num_states,
        observation_dim,
        action_dim,
        hidden_dim=64,
        belief_dim=64,
        latent_dim=64,
        learning_rate=1e-3,
        discount_factor=0.99,
        entropy_weight=0.01,
        kl_weight=0.01,
        target_update_rate=0.005,
        device=None,
        buffer_capacity=1000,
        max_trajectory_length=50,
        use_si=False,
        si_importance=100.0,
        si_damping=0.1,
        si_exclude_final_layers=False,
        continuous_actions=False,
        num_gnn_layers=2,
        num_attn_heads=4,
        temporal_window_size=5,
    ):

        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()

        self.agent_id = agent_id
        self.num_agents = num_agents
        self.num_states = num_states
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.target_update_rate = target_update_rate
        self.device = device
        self.buffer_capacity = buffer_capacity
        self.max_trajectory_length = max_trajectory_length
        self.use_si = use_si
        self.si_importance = si_importance
        self.si_damping = si_damping
        self.si_exclude_final_layers = si_exclude_final_layers
        self.continuous_actions = continuous_actions

        # Global variables for action logits and neighbor action logits
        self.action_logits = None
        self.neighbor_action_logits = None

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            observation_dim=observation_dim,
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            device=device,
            sequence_length=max_trajectory_length,
        )

        # Create belief processor
        self.belief_processor = TransformerBeliefProcessor(
            hidden_dim=belief_dim,
            action_dim=action_dim,
            device=device,
            num_belief_states=num_states,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        ).to(device)

        # Create GNN-based inference module (always used now)
        self.inference_module = TemporalGNN(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            num_agents=num_agents,
            device=device,
            num_belief_states=num_states,
            num_gnn_layers=num_gnn_layers,
            num_attn_heads=num_attn_heads,
            dropout=0.1,
            temporal_window_size=temporal_window_size,
        ).to(device)

        # Use appropriate policy network based on action space
        if continuous_actions:
            self.policy = ContinuousPolicyNetwork(
                belief_dim=belief_dim,
                latent_dim=latent_dim,
                action_dim=1,  # For strategic experimentation, just one value between 0 and 1
                hidden_dim=hidden_dim,
                device=device,
                min_action=0.0,
                max_action=1.0,
            ).to(device)
        else:
            self.policy = PolicyNetwork(
                belief_dim=belief_dim,
                latent_dim=latent_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device,
            ).to(device)

        self.q_network1 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device,
        ).to(device)

        self.q_network2 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device,
        ).to(device)

        # Create target networks
        self.target_q_network1 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device,
        ).to(device)

        self.target_q_network2 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device,
        ).to(device)

        # Copy parameters to target networks
        self.target_q_network1.load_state_dict(self.q_network1.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())

        # Average reward estimate (for average reward formulation)
        self.gain_parameter = nn.Parameter(torch.tensor(0.0, device=device))

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )

        # Separate Transformer optimizer - only for belief processor
        self.transformer_optimizer = torch.optim.Adam(
            self.belief_processor.parameters(), lr=learning_rate
        )

        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            lr=learning_rate,
        )

        # Set up inference optimizer based on which inference module we're using
        self.inference_optimizer = torch.optim.Adam(
            self.inference_module.parameters(), lr=learning_rate
        )

        self.gain_optimizer = torch.optim.Adam([self.gain_parameter], lr=learning_rate)

        # Initialize belief and latent states with correct shapes
        self.current_belief = (
            torch.ones(1, 1, belief_dim, device=device)
            / self.belief_processor.hidden_dim
        )  # [1, batch_size=1, hidden_dim]
        self.current_latent = (
            torch.ones(1, latent_dim, device=device) / latent_dim
        )  # [1, latent_dim]
        self.current_mean = torch.zeros(1, latent_dim, device=device)
        self.current_logvar = torch.zeros(1, latent_dim, device=device)

        # Initialize belief distribution
        self.current_belief_distribution = (
            torch.ones(1, self.belief_processor.num_belief_states, device=device)
            / self.belief_processor.num_belief_states
        )

        # Initialize opponent belief distribution
        self.current_opponent_belief_distribution = (
            torch.ones(1, self.num_agents, device=device) / self.num_agents
        )

        # For tracking learning metrics
        self.action_probs_history = []

        # Episode tracking
        self.episode_step = 0

        # SI setup
        if self.use_si:
            # Use a much higher importance factor for better protection against forgetting

            # Define layers to exclude from SI if requested
            self.excluded_belief_layers = []
            self.excluded_policy_layers = []

            if si_exclude_final_layers:
                # Define final layers to exclude from SI protection
                self.excluded_belief_layers = ["belief_head.weight", "belief_head.bias"]
                self.excluded_policy_layers = [
                    "fc3.weight",
                    "fc3.bias",
                ]  # Final classification layer
                print(
                    f"Agent {agent_id}: Excluding final layers from SI protection: {self.excluded_belief_layers + self.excluded_policy_layers}"
                )

            # Initialize SI for belief processor with layer exclusion
            self.belief_si = SILoss(
                model=self.belief_processor,
                importance=si_importance,
                damping=si_damping,
                device=device,
                excluded_layers=self.excluded_belief_layers,
            )

            # Initialize SI for policy network with layer exclusion
            self.policy_si = SILoss(
                model=self.policy,
                importance=si_importance,
                damping=si_damping,
                device=device,
                excluded_layers=self.excluded_policy_layers,
            )

            # Track previously seen true states
            self.seen_true_states = set()
            self.path_integrals_calculated = False

            # Track SI trackers for each state to enable comparison across tasks
            self.state_belief_si_trackers = {}
            self.state_policy_si_trackers = {}

            # Track current active state
            self.current_true_state = None

            # Add counter for debugging
            self.si_debug_counter = 0

    def observe(self, signal, neighbor_actions):
        """Update belief state based on new observation."""
        # Check if this is the first observation of the episode
        is_first_obs = self.episode_step == 0
        self.episode_step += 1

        # Pass only signal and belief to the belief processor (ignoring neighbor_actions)
        belief, belief_distribution = self.belief_processor(
            signal, current_belief=self.current_belief
        )

        # Store belief state with consistent shape [1, batch_size=1, hidden_dim]
        self.current_belief = belief

        # Store the belief distribution
        self.current_belief_distribution = belief_distribution

        return self.current_belief, self.current_belief_distribution

    def infer_latent(self, signal, neighbor_actions, reward, next_signal):
        """Infer latent state of neighbors based on our observations which already contain neighbor actions."""

        # Use the GNN for inference
        mean, logvar, opponent_belief_distribution = self.inference_module(
            signal, neighbor_actions, reward, next_signal, self.current_latent
        )

        # Sample based on reparameterized distribution
        # Ref: https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
        # Add numerical stability safeguards
        # First, clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)

        # Then calculate variance with better safety measures
        var = torch.exp(0.5 * logvar)
        epsilon = 1e-6
        var = torch.clamp(var, min=epsilon, max=1e6)  # Also add maximum bound
        distribution = torch.distributions.Normal(mean, var)
        new_latent = distribution.rsample()

        # Store the current latent, mean, logvar, and opponent belief distribution
        self.current_latent = new_latent.unsqueeze(0)
        self.current_mean = mean
        self.current_logvar = logvar
        self.current_opponent_belief_distribution = opponent_belief_distribution

        return new_latent

    def store_transition(
        self,
        observation,
        belief,
        latent,
        action,
        reward,
        next_observation,
        next_belief,
        next_latent,
        mean=None,
        logvar=None,
        neighbor_actions=None,
    ):
        """Store a transition in the replay buffer."""
        # Ensure belief states have consistent shape before storing
        belief = self.belief_processor.standardize_belief_state(belief)
        next_belief = self.belief_processor.standardize_belief_state(next_belief)

        self.replay_buffer.push(
            observation,
            belief,
            latent,
            action,
            reward,
            next_observation,
            next_belief,
            next_latent,
            mean,
            logvar,
            neighbor_actions,
        )

    def set_train_mode(self):
        """Set all networks to training mode."""
        self.belief_processor.train()
        self.inference_module.train()
        self.policy.train()
        self.q_network1.train()
        self.q_network2.train()
        self.target_q_network1.train()
        self.target_q_network2.train()

    def set_eval_mode(self):
        """Set all networks to evaluation mode."""
        self.belief_processor.eval()
        self.inference_module.eval()
        self.policy.eval()
        self.q_network1.eval()
        self.q_network2.eval()
        self.target_q_network1.eval()
        self.target_q_network2.eval()

    def reset_internal_state(self):
        """Reset the agent's internal state (belief and latent variables)."""
        # Use zeros for a complete reset with correct shapes
        self.current_belief = torch.zeros(
            1, 1, self.belief_processor.hidden_dim, device=self.device
        )  # [1, batch_size=1, hidden_dim]
        self.current_latent = torch.zeros(1, self.latent_dim, device=self.device)
        self.current_mean = torch.zeros(1, self.latent_dim, device=self.device)
        self.current_logvar = torch.zeros(1, self.latent_dim, device=self.device)

        # Reset belief distribution
        self.current_belief_distribution = (
            torch.ones(1, self.belief_processor.num_belief_states, device=self.device)
            / self.belief_processor.num_belief_states
        )

        # Detach all tensors to ensure no gradient flow between episodes
        self.current_belief = self.current_belief.detach()
        self.current_latent = self.current_latent.detach()
        self.current_mean = self.current_mean.detach()
        self.current_logvar = self.current_logvar.detach()
        if self.current_belief_distribution is not None:
            self.current_belief_distribution = self.current_belief_distribution.detach()
        if (
            hasattr(self, "current_opponent_belief_distribution")
            and self.current_opponent_belief_distribution is not None
        ):
            self.current_opponent_belief_distribution = (
                self.current_opponent_belief_distribution.detach()
            )

        # If using GNN, reset its temporal memory
        self.inference_module.reset_memory()

    def select_action(self):
        """Select action based on current belief and latent."""
        if self.continuous_actions:
            # For continuous actions (strategic experimentation)
            action = self.policy.get_action(
                self.current_belief, self.current_latent
            )

            # Convert to numpy and scalar (for 1D action space)
            action_value = action.squeeze().detach().cpu().numpy()

            # Return the action value directly (allocation between 0 and 1)
            return action_value.item(), np.array([action_value.item()])
        else:
            # For discrete actions (original approach)
            # Calculate fresh action logits for action selection
            action_logits = self.policy(self.current_belief, self.current_latent)
        
            # Store a detached copy for caching
            self.action_logits = action_logits.detach()

            # Convert to probabilities
            action_probs = F.softmax(action_logits, dim=-1)

            # Store probability of incorrect action for learning rate calculation
            self.action_probs_history.append(
                action_probs.squeeze(0).detach().cpu().numpy()
            )

            # Sample action from distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()

            return action, action_probs.squeeze(0).detach().cpu().numpy()

    def train(self, batch_size=32, sequence_length=32):
        """Train the agent using sequential data from the replay buffer."""
        # Sample sequential data from the replay buffer
        batch_sequences = self.replay_buffer.sample(
            batch_size, sequence_length, mode="sequence"
        )

        # Update networks using sequential data
        return self.update(batch_sequences)

    def update(self, batch_sequences):
        """Update networks using batched data."""
        # Initialize losses
        total_inference_loss = 0
        total_critic_loss = 0
        total_policy_loss = 0
        total_transformer_loss = 0

        # Check if we have a single batch or a list of sequences
        if isinstance(batch_sequences, tuple):
            # Single batch case
            batch_sequences = [batch_sequences]

        # Process each time step in the sequence
        for t, batch in enumerate(batch_sequences):
            # Unpack the batch
            (
                signals,
                neighbor_actions,
                beliefs,
                latents,
                actions,
                rewards,
                next_signals,
                next_beliefs,
                next_latents,
                means,
                logvars,
            ) = batch

            # Update inference module
            inference_loss = self._update_inference(
                signals, neighbor_actions, next_signals, next_latents, means, logvars
            )
            total_inference_loss += inference_loss

            # Update policy (with advantage for GRU)
            policy_result = self._update_policy(
                beliefs, latents, actions, neighbor_actions
            )
            policy_loss, _ = policy_result
            total_policy_loss += policy_loss

            # Update Transformer with advantage
            transformer_loss = self._update_transformer(
                signals, neighbor_actions, beliefs, next_signals
            )
            total_transformer_loss += transformer_loss

            # Update Q-networks
            critic_loss = self._update_critics(
                signals,
                neighbor_actions,
                beliefs,
                latents,
                actions,
                neighbor_actions,
                rewards,
                next_signals,
                next_beliefs,
                next_latents,
            )
            total_critic_loss += critic_loss

        # Update target networks once per sequence
        self._update_targets()

        # Return average losses
        sequence_length = len(batch_sequences)
        return {
            "inference_loss": total_inference_loss / sequence_length,
            "critic_loss": total_critic_loss / sequence_length,
            "policy_loss": total_policy_loss / sequence_length,
            "transformer_loss": total_transformer_loss / sequence_length,
        }

    def _update_inference(
        self, signals, neighbor_actions, next_signals, next_latents, means, logvars
    ):
        """Update inference module with FURTHER-style temporal KL."""

        # We need dummy rewards for the GNN forward pass
        batch_size = signals.size(0)
        dummy_rewards = torch.zeros(batch_size, 1, device=self.device)

        # Forward pass through GNN to get new distribution parameters
        # Note: we detach next_latents to avoid gradients flowing back through the target network
        new_means, new_logvars, _ = self.inference_module(
            signals, neighbor_actions, dummy_rewards, next_signals, next_latents.detach()
        )

        # Generate action predictions using the current batch
        batch_neighbor_logits = self.inference_module.predict_actions(
            signals, next_latents.detach()
        )

        # Reshape batch_neighbor_logits if needed for cross entropy
        if batch_neighbor_logits.dim() == 3:
            batch_size, seq_len, action_dim = batch_neighbor_logits.shape
            batch_neighbor_logits = batch_neighbor_logits.view(
                batch_size * seq_len, action_dim
            )
            neighbor_actions_reshaped = neighbor_actions.view(-1)
        else:
            neighbor_actions_reshaped = neighbor_actions

        # Calculate reconstruction loss
        recon_loss = F.cross_entropy(
            batch_neighbor_logits, neighbor_actions_reshaped
        )

        # Calculate temporal KL divergence with numerical stability
        kl_loss = self._calculate_temporal_kl_divergence(new_means, new_logvars)

        # Total loss
        inference_loss = recon_loss + kl_loss

        # Update networks
        self.inference_optimizer.zero_grad()
        inference_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.inference_module.parameters(), max_norm=1.0
        )
        self.inference_optimizer.step()

        return inference_loss.item()

    def _calculate_temporal_kl_divergence(self, means_seq, logvars_seq):
        """Calculate KL divergence between sequential latent states (temporal smoothing)."""

        # KL(N(mu,E), N(m, S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m - mu)^T S^-1 (m - mu)))
        # Ref: https://github.com/lmzintgraf/varibad/blob/master/vae.py
        # Ref: https://github.com/dkkim93/further/blob/main/algorithm/further/agent.py

        kl_first_term = torch.sum(logvars_seq[:-1, :], dim=-1) - torch.sum(
            logvars_seq[1:, :], dim=-1
        )
        kl_second_term = self.latent_dim
        kl_third_term = torch.sum(
            1.0 / torch.exp(logvars_seq[:-1, :]) * torch.exp(logvars_seq[1:, :]), dim=-1
        )
        kl_fourth_term = (
            (means_seq[:-1, :] - means_seq[1:, :])
            / torch.exp(logvars_seq[:-1, :])
            * (means_seq[:-1, :] - means_seq[1:, :])
        )
        kl_fourth_term = kl_fourth_term.sum(dim=-1)

        kl = 0.5 * (kl_first_term - kl_second_term + kl_third_term + kl_fourth_term)

        return self.kl_weight * torch.mean(kl)

    def _update_critics(
        self,
        signals,
        neighbor_actions,
        beliefs,
        latents,
        actions,
        next_neighbor_actions,
        rewards,
        next_signals,
        next_beliefs,
        next_latents,
    ):
        """Update Q-networks."""
        if self.continuous_actions:
            # For continuous actions, we need to handle the Q-networks differently
            # Convert actions to the correct format for Q-network input
            if isinstance(actions, torch.Tensor):
                if actions.dim() == 1:
                    continuous_actions = actions.unsqueeze(1).float()
                else:
                    continuous_actions = actions.float()
            else:
                continuous_actions = torch.tensor(
                    [[float(actions)]], device=self.device
                )

            # Get current Q-values
            current_q1 = self.q_network1(beliefs, latents, neighbor_actions)
            current_q2 = self.q_network2(beliefs, latents, neighbor_actions)

            # For continuous case, we use the mean Q-value as our estimate
            q1 = current_q1.mean(dim=1, keepdim=True)
            q2 = current_q2.mean(dim=1, keepdim=True)

            # Compute next action distribution
            with torch.no_grad():
                # Get allocation from the policy
                next_allocation = self.policy(next_beliefs, next_latents)

                # Compute Q-values with target networks
                next_q1 = self.target_q_network1(
                    next_beliefs, next_latents, next_neighbor_actions
                )
                next_q2 = self.target_q_network2(
                    next_beliefs, next_latents, next_neighbor_actions
                )

                # Take minimum Q-value
                next_q = torch.min(next_q1, next_q2)

                # For continuous case, use the mean Q-value
                expected_q = next_q.mean(dim=1, keepdim=True)

                # Compute target (no entropy term since deterministic policy)
                if self.discount_factor > 0:  # Discounted return
                    target_q = rewards + self.discount_factor * expected_q
                else:  # Average reward
                    target_q = rewards - self.gain_parameter + expected_q
        else:
            # Original discrete action implementation
            # Get current Q-values
            q1 = self.q_network1(beliefs, latents, neighbor_actions).gather(
                1, actions.unsqueeze(1)
            )
            q2 = self.q_network2(beliefs, latents, neighbor_actions).gather(
                1, actions.unsqueeze(1)
            )

            # Compute next action probabilities
            with torch.no_grad():
                # Calculate fresh action logits for critic update
                next_action_logits = self.policy(next_beliefs, next_latents)
                next_action_probs = F.softmax(next_action_logits, dim=1)
                next_log_probs = F.log_softmax(next_action_logits, dim=1)
                entropy = -torch.sum(
                    next_action_probs * next_log_probs, dim=1, keepdim=True
                )

                # Compute Q-values with predicted next neighbor actions
                next_q1 = self.target_q_network1(
                    next_beliefs, next_latents, next_neighbor_actions
                )
                next_q2 = self.target_q_network2(
                    next_beliefs, next_latents, next_neighbor_actions
                )

                # Take minimum
                next_q = torch.min(next_q1, next_q2)

                # Expected Q-value
                expected_q = (next_action_probs * next_q).sum(dim=1, keepdim=True)

                # Add entropy
                expected_q = expected_q + self.entropy_weight * entropy

                # Compute target
                if self.discount_factor > 0:  # Discounted return
                    target_q = rewards + self.discount_factor * expected_q
                else:  # Average reward
                    target_q = rewards - self.gain_parameter + expected_q

        # Compute loss
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        # Update networks
        self.q_optimizer.zero_grad()
        if self.discount_factor == 0:  # Only update gain parameter for average reward
            self.gain_optimizer.zero_grad()

        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            max_norm=1.0,
        )
        self.q_optimizer.step()

        if self.discount_factor == 0:
            self.gain_optimizer.step()

        return q_loss.item()

    def _update_policy(self, beliefs, latents, actions, neighbor_actions):
        """Update policy network and calculate advantage for Transformer training.

        Returns:
            Tuple of (policy_loss_value, advantage)
        """
        if self.continuous_actions:
            # For continuous actions, policy now outputs allocation directly
            allocation = self.policy(beliefs, latents)

            # For continuous case, actions are the allocation values
            if isinstance(actions, torch.Tensor):
                if actions.dim() == 1:
                    # Reshape to [batch_size, 1] for proper handling
                    continuous_actions = actions.unsqueeze(1).float()
                else:
                    continuous_actions = actions.float()
            else:
                # Handle scalar actions
                continuous_actions = torch.tensor(
                    [[float(actions)]], device=self.device
                )

            # Calculate MSE loss between predicted and actual allocations
            policy_loss = F.mse_loss(allocation, continuous_actions)

            # Calculate advantage for Transformer training
            with torch.no_grad():
                # Get Q-values for the actions
                q1 = self.q_network1(beliefs, latents, neighbor_actions)
                q2 = self.q_network2(beliefs, latents, neighbor_actions)
                q = torch.min(q1, q2)

                # For continuous actions, we use the taken action's Q-value as advantage
                advantage = q.mean()
        else:
            # Generate fresh action logits for the batch
            action_logits = self.policy(beliefs, latents)

            # Calculate probabilities
            action_probs = F.softmax(action_logits, dim=1)
            log_probs = F.log_softmax(action_logits, dim=1)

            # Calculate entropy
            entropy = -torch.sum(action_probs * log_probs, dim=1).mean()

            # Get the log probability of the taken actions
            indices = actions.long().unsqueeze(1)
            taken_log_probs = torch.gather(log_probs, 1, indices).squeeze(1)

            # Get Q-values for advantage
            q1 = self.q_network1(beliefs, latents, neighbor_actions)
            q2 = self.q_network2(beliefs, latents, neighbor_actions)
            q = torch.min(q1, q2)

            # Calculate expected values of the current policy
            v = torch.sum(action_probs * q, dim=1)

            # Get the Q-values of the taken actions
            q_taken = torch.gather(q, 1, indices).squeeze(1)

            # Calculate advantages A(s, a) = Q(s, a) - V(s)
            advantage = q_taken - v

            # Policy gradient loss
            policy_loss = -(taken_log_probs * advantage.detach()).mean()

            # Add entropy term for exploration
            policy_loss = policy_loss - self.entropy_weight * entropy

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        # Update SI trajectory before optimizer step
        if self.use_si:
            self.policy_si.update_trajectory()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Update SI trajectory after optimizer step
        if self.use_si:
            self.policy_si.update_trajectory_post_step()

        return policy_loss.item(), advantage

    def _update_transformer(self, signals, neighbor_actions, beliefs, next_signals):
        """
        Update belief processor (Transformer) by optimizing for next state prediction.

        Args:
            signals: Current signals/observations
            neighbor_actions: Current neighbor actions (ignored in signal-only mode)
            beliefs: Current belief states
            next_signals: Next signals/observations

        Returns:
            float: Loss value
        """
        # Process current signals through Transformer to get next belief (ignoring neighbor_actions)
        _, belief_distributions = self.belief_processor(signals, current_belief=beliefs)

        # Detect if signal is continuous (1-dimensional) or discrete (one-hot encoded)
        is_continuous_signal = next_signals.size(1) == 1

        # The belief distribution should predict the next observation (signal)
        if is_continuous_signal:
            # For continuous signals, we use MSE loss directly on the raw values
            transformer_loss = F.mse_loss(
                belief_distributions, next_signals, reduction="mean"
            )
        else:
            # For discrete signals, use binary cross entropy
            transformer_loss = (
                F.binary_cross_entropy(
                    belief_distributions, next_signals, reduction="none"
                )
                .sum(dim=1)
                .mean()
            )

        # Add SI loss if enabled
        si_loss = 0.0
        if self.use_si and self.path_integrals_calculated:
            si_loss = self.belief_si.calculate_loss()
            transformer_loss += si_loss

        # Update Transformer parameters
        self.transformer_optimizer.zero_grad()
        transformer_loss.backward()

        # Update SI trajectory before optimizer step
        if self.use_si:
            self.belief_si.update_trajectory()

        torch.nn.utils.clip_grad_norm_(self.belief_processor.parameters(), max_norm=1.0)
        self.transformer_optimizer.step()

        # Update SI trajectory after optimizer step
        if self.use_si:
            self.belief_si.update_trajectory_post_step()

        return transformer_loss.item()

    def _update_targets(self):
        """Update target networks."""
        for target_param, param in zip(
            self.target_q_network1.parameters(), self.q_network1.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_rate)
                + param.data * self.target_update_rate
            )

        for target_param, param in zip(
            self.target_q_network2.parameters(), self.q_network2.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_rate)
                + param.data * self.target_update_rate
            )

    def save(self, path):
        """Save the agent's networks to a file."""
        checkpoint = {
            "belief_processor": self.belief_processor.state_dict(),
            "policy": self.policy.state_dict(),
            "q_network1": self.q_network1.state_dict(),
            "q_network2": self.q_network2.state_dict(),
            "target_q_network1": self.target_q_network1.state_dict(),
            "target_q_network2": self.target_q_network2.state_dict(),
            "gain_parameter": self.gain_parameter,
            "use_si": self.use_si,
        }

        # Save the inference module
        checkpoint["inference_module"] = self.inference_module.state_dict()

        torch.save(checkpoint, path)

    def load(self, path, evaluation_mode=False):
        """
        Load agent state from a checkpoint file.

        Args:
            path: Path to the checkpoint file
            evaluation_mode: Whether to set the model in evaluation mode after loading
            
        Raises:
            ArchitectureMismatchError: When saved model architecture doesn't match current
            ModelComponentError: When model components cannot be loaded
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Check if we're loading a GRU model into a Transformer model
        is_gru_to_transformer = False
        try:
            # Try to load belief processor (may fail if architecture changed from GRU to Transformer)
            self.belief_processor.load_state_dict(checkpoint["belief_processor"])
        except RuntimeError as e:
            raise ArchitectureMismatchError(
                f"Could not load belief processor due to architecture change: {e}. "
                f"The saved model likely uses a different belief processor architecture "
                f"(e.g., GRU vs Transformer). Please ensure model architectures match."
            )

        # Check for architecture mismatch between SI and non-SI
        use_si_in_checkpoint = checkpoint.get("use_si", False)
        if use_si_in_checkpoint != self.use_si:
            raise ArchitectureMismatchError(
                f"Architecture mismatch detected. Saved model uses {'SI' if use_si_in_checkpoint else 'no SI'} "
                f"but current agent uses {'SI' if self.use_si else 'no SI'}. "
                f"Please initialize the agent with use_si={use_si_in_checkpoint} to match the saved model."
            )

        # Load core components that should be compatible
        try:
            self.inference_module.load_state_dict(checkpoint["inference_module"])
        except RuntimeError as e:
            raise ModelComponentError(f"Could not load inference module state: {e}")

        # Always try to load policy network
        try:
            self.policy.load_state_dict(checkpoint["policy"])
        except RuntimeError as e:
            raise ModelComponentError(f"Could not load policy network: {e}")

        # Handle Q-networks
        try:
            self.q_network1.load_state_dict(checkpoint["q_network1"])
            self.q_network2.load_state_dict(checkpoint["q_network2"])
            self.target_q_network1.load_state_dict(checkpoint["target_q_network1"])
            self.target_q_network2.load_state_dict(checkpoint["target_q_network2"])
        except RuntimeError as e:
            raise ModelComponentError(
                f"Could not load Q-networks due to architecture changes: {e}. "
                f"This indicates a significant architecture change that requires retraining."
            )

        try:
            self.gain_parameter.data = checkpoint["gain_parameter"]
        except KeyError:
            raise ModelComponentError("Gain parameter not found in checkpoint")
        except Exception as e:
            raise ModelComponentError(f"Could not load gain parameter: {e}")

        # Reset internal state after loading
        self.reset_internal_state()

        # Set the model to evaluation mode if requested
        if evaluation_mode:
            self.set_eval_mode()
            print(f"Model set to evaluation mode.")
        else:
            self.set_train_mode()
            print(f"Model set to training mode.")

    def get_belief_state(self):
        """Return the current belief state.

        Returns:
            belief: Current belief state tensor with shape [1, batch_size=1, hidden_dim]
        """
        return self.current_belief

    def get_latent_state(self):
        """Return the current latent state.

        Returns:
            latent: Current latent state tensor
        """
        return self.current_latent

    def get_belief_distribution(self):
        """Return the current belief distribution.

        Returns:
            belief_distribution: Current belief distribution tensor or None if not available
        """
        return self.current_belief_distribution

    def get_latent_distribution_params(self):
        """Return the current latent distribution parameters (mean and logvar).

        Returns:
            mean: Current mean of the latent distribution
            logvar: Current log variance of the latent distribution
        """
        return self.current_mean, self.current_logvar

    def get_opponent_belief_distribution(self):
        """Return the current opponent belief distribution.

        Returns:
            opponent_belief_distribution: Current opponent belief distribution tensor or None if not available
        """
        return (
            self.current_opponent_belief_distribution
            if hasattr(self, "current_opponent_belief_distribution")
            else None
        )

    def calculate_path_integrals(self, replay_buffer):
        """
        Legacy method for compatibility. Use register_task() for online tracking instead.
        This method may be called when transitioning to a new task in simulation.

        Args:
            replay_buffer: The replay buffer containing transitions (not used)
        """
        if not self.use_si:
            print(f"Agent {self.agent_id}: SI not enabled. Skipping.")
            return

        print(
            f"Agent {self.agent_id}: Using online path integral tracking instead of replay buffer."
        )

        # Register the current task if not done already
        if hasattr(self, "belief_si") and hasattr(self, "policy_si"):
            if self.current_true_state is not None:
                print(f"Registering task for true state {self.current_true_state}")
                self.belief_si.register_task()
                self.policy_si.register_task()

            # Create copies of the trackers for visualization and comparison
            if (
                hasattr(self, "current_true_state")
                and self.current_true_state is not None
            ):
                print(f"Storing SI trackers for true state {self.current_true_state}")
                self.state_belief_si_trackers[self.current_true_state] = (
                    self._clone_si_tracker(self.belief_si)
                )
                self.state_policy_si_trackers[self.current_true_state] = (
                    self._clone_si_tracker(self.policy_si)
                )

        # Mark that this agent is using SI
        self.path_integrals_calculated = True
        print(f"Agent {self.agent_id}: Online SI tracking enabled.")

    def _clone_si_tracker(self, tracker):
        """Create a deep copy of an SI tracker for state comparison."""
        # Create a new tracker with the same parameters
        cloned_tracker = SILoss(
            model=tracker.model,
            importance=tracker.importance,
            damping=tracker.damping,
            device=tracker.device,
        )

        # Copy importance scores (these need to be deep copied)
        for name, importance in tracker.importance_scores.items():
            cloned_tracker.importance_scores[name] = importance.clone().detach()

        # Copy previous parameters (reference parameters)
        if hasattr(tracker, "previous_params"):
            for name, param in tracker.previous_params.items():
                cloned_tracker.previous_params[name] = param.clone().detach()

        # Copy path integrals if they exist
        if hasattr(tracker, "param_path_integrals"):
            for name, integral in tracker.param_path_integrals.items():
                cloned_tracker.param_path_integrals[name] = integral.clone().detach()

        return cloned_tracker
