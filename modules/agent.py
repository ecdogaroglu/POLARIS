import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.networks import EncoderNetwork, DecoderNetwork, PolicyNetwork, QNetwork, TransformerBeliefProcessor, TemporalGNN
from modules.replay_buffer import ReplayBuffer
from modules.utils import get_best_device, encode_observation
from modules.si import SILoss, calculate_path_integral_from_replay_buffer

class POLARISAgent:
    """POLARIS agent for social learning with additional advantage-based Transformer training."""
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
        use_gnn=True,
        use_si=False,
        si_importance=100.0,
        si_damping=0.1,
        si_exclude_final_layers=False
    ):
        
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        print(f"Using device: {device}")
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.num_states = num_states
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.target_update_rate = target_update_rate
        self.max_trajectory_length = max_trajectory_length
        self.latent_dim = latent_dim
        self.use_gnn = use_gnn
        
        # Global variables for action logits and neighbor action logits
        self.action_logits = None
        self.neighbor_action_logits = None
        
        # Initialize replay buffer with our enhanced version
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            observation_dim=observation_dim,
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            device=device,
            sequence_length=max_trajectory_length
        )
        
        # Initialize all networks
        self.belief_processor = TransformerBeliefProcessor(
            hidden_dim=belief_dim,
            action_dim=action_dim,
            device=device,
            num_belief_states=num_states,
            nhead=4,  # Number of attention heads
            num_layers=2,  # Number of transformer layers
            dropout=0.1  # Dropout rate
        ).to(device)
        
        # Initialize either the GNN or the traditional encoder-decoder
        if self.use_gnn:
            # Use the new TemporalGNN for inference learning
            self.inference_module = TemporalGNN(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                num_agents=num_agents,
                device=device,
                num_belief_states=num_states,
                num_gnn_layers=2,  # Default value, will be updated later if needed
                num_attn_heads=4,  # Default value, will be updated later if needed
                dropout=0.1,
                temporal_window_size=5  # Default value, will be updated later if needed
            ).to(device)
        else:
            # Use the traditional encoder-decoder approach
            self.encoder = EncoderNetwork(
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_agents=num_agents,
                device=device,
                num_belief_states=num_states
            ).to(device)
            
            self.decoder = DecoderNetwork(
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_agents=num_agents,
                num_belief_states=num_states,
                device=device
            ).to(device)
        
        self.policy = PolicyNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        ).to(device)
        
        self.q_network1 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        self.q_network2 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        # Create target networks
        self.target_q_network1 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        self.target_q_network2 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        # Copy parameters to target networks
        self.target_q_network1.load_state_dict(self.q_network1.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())
        
        # Average reward estimate (for average reward formulation)
        self.gain_parameter = nn.Parameter(torch.tensor(0.0, device=device))
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )
        
        # Separate Transformer optimizer - only for belief processor
        self.transformer_optimizer = torch.optim.Adam(
            self.belief_processor.parameters(),
            lr=learning_rate
        )
        
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + 
            list(self.q_network2.parameters()),
            lr=learning_rate
        )
        
        # Set up inference optimizer based on which inference module we're using
        if self.use_gnn:
            self.inference_optimizer = torch.optim.Adam(
                self.inference_module.parameters(),
                lr=learning_rate
            )
        else:
            self.inference_optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=learning_rate
            )
            
        self.gain_optimizer = torch.optim.Adam([self.gain_parameter], lr=learning_rate)
        
        # Initialize belief and latent states with correct shapes
        self.current_belief = torch.ones(1, 1, belief_dim, device=device) / self.belief_processor.hidden_dim  # [1, batch_size=1, hidden_dim]
        self.current_latent = torch.ones(1, latent_dim, device=device) / latent_dim  # [1, latent_dim]
        self.current_mean = torch.zeros(1, latent_dim, device=device)
        self.current_logvar = torch.zeros(1, latent_dim, device=device)
        
        # Initialize belief distribution
        self.current_belief_distribution = torch.ones(1, self.belief_processor.num_belief_states, device=device) / self.belief_processor.num_belief_states
        
        # Initialize opponent belief distribution
        self.current_opponent_belief_distribution = torch.ones(1, self.num_agents, device=device) / self.num_agents

        # For tracking learning metrics
        self.action_probs_history = []
        
        # Episode tracking
        self.episode_step = 0
        
        # SI setup
        self.use_si = use_si
        self.si_importance = si_importance
        self.si_damping = si_damping
        self.si_exclude_final_layers = si_exclude_final_layers
        
        if self.use_si:
            # Use a much higher importance factor for better protection against forgetting
            # The default value might be too low for this specific task
            actual_importance = si_importance * 10.0  # Increase by 10x
            
            # Define layers to exclude from SI if requested
            self.excluded_belief_layers = []
            self.excluded_policy_layers = []
            
            if si_exclude_final_layers:
                # Define final layers to exclude from SI protection
                self.excluded_belief_layers = ['belief_head.weight', 'belief_head.bias']
                self.excluded_policy_layers = ['fc3.weight', 'fc3.bias']  # Final classification layer
                print(f"Agent {agent_id}: Excluding final layers from SI protection: {self.excluded_belief_layers + self.excluded_policy_layers}")
            
            # Initialize SI for belief processor with layer exclusion
            self.belief_si = SILoss(
                model=self.belief_processor,
                importance=actual_importance,
                damping=si_damping,
                exclude_layers=self.excluded_belief_layers,
                device=device
            )
            
            # Initialize SI for policy network with layer exclusion
            self.policy_si = SILoss(
                model=self.policy,
                importance=actual_importance,
                damping=si_damping,
                exclude_layers=self.excluded_policy_layers,
                device=device
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
            
            # For tracking parameter changes during training
            self.belief_si_step_count = 0
            self.policy_si_step_count = 0
            self.accumulated_belief_loss = 0
            self.accumulated_policy_loss = 0
            self.si_accumulated_steps = 0
    
    def observe(self, signal, neighbor_actions):
        """Update belief state based on new observation."""
        # Check if this is the first observation of the episode
        is_first_obs = (self.episode_step == 0)
        self.episode_step += 1

        # Pass observation and belief to the belief processor
        belief, belief_distribution = self.belief_processor(
            signal,
            neighbor_actions, 
            self.current_belief
        )
        
        # Store belief state with consistent shape [1, batch_size=1, hidden_dim]
        self.current_belief = belief
        
        # Store the belief distribution
        self.current_belief_distribution = belief_distribution
        
        return self.current_belief, self.current_belief_distribution
        
    def store_transition(self, observation, belief, latent, action, reward, 
                         next_observation, next_belief, next_latent, mean=None, logvar=None, neighbor_actions=None):
        """Store a transition in the replay buffer."""
        # Ensure belief states have consistent shape before storing
        belief = self.belief_processor.standardize_belief_state(belief)
        next_belief = self.belief_processor.standardize_belief_state(next_belief)
        
        self.replay_buffer.push(
            observation, belief, latent, action, reward,
            next_observation, next_belief, next_latent, mean, logvar, neighbor_actions
        )
        
    def set_train_mode(self):
        """Set all networks to training mode."""
        self.belief_processor.train()
        if self.use_gnn:
            self.inference_module.train()
        else:
            self.encoder.train()
            self.decoder.train()
        self.policy.train()
        self.q_network1.train()
        self.q_network2.train()
        self.target_q_network1.train()
        self.target_q_network2.train()
        
    def set_eval_mode(self):
        """Set all networks to evaluation mode."""
        self.belief_processor.eval()
        if self.use_gnn:
            self.inference_module.eval()
        else:
            self.encoder.eval()
            self.decoder.eval()
        self.policy.eval()
        self.q_network1.eval()
        self.q_network2.eval()
        self.target_q_network1.eval()
        self.target_q_network2.eval()
        
    def reset_internal_state(self):
        """Reset internal states to initial values."""
        # Reset belief and latent states
        self.current_belief = torch.ones(1, 1, self.belief_processor.hidden_dim, device=self.device) / self.belief_processor.hidden_dim
        self.current_latent = torch.ones(1, self.latent_dim, device=self.device) / self.latent_dim
        self.current_mean = torch.zeros(1, self.latent_dim, device=self.device)
        self.current_logvar = torch.zeros(1, self.latent_dim, device=self.device)
        
        # Reset belief probability distribution
        self.current_belief_distribution = torch.ones(1, self.belief_processor.num_belief_states, device=self.device) / self.belief_processor.num_belief_states
        
        # Reset opponent belief distribution
        self.current_opponent_belief_distribution = torch.ones(1, self.num_agents, device=self.device) / self.num_agents
        
        # Reset action probabilities history
        self.action_probs_history = []
        
        # Reset episode step counter
        self.episode_step = 0
        
        # Reset SI trackers if using SI and not already initialized
        if self.use_si:
            # Check if SI trackers are already initialized
            if not hasattr(self, 'belief_si') or self.belief_si is None:
                print(f"Agent {self.agent_id}: Initializing SI trackers for belief processor")
                self.belief_si = SILoss(
                    model=self.belief_processor,
                    importance=self.si_importance,
                    damping=self.si_damping,
                    exclude_layers=self.excluded_belief_layers if hasattr(self, 'excluded_belief_layers') else [],
                    device=self.device
                )
            
            if not hasattr(self, 'policy_si') or self.policy_si is None:
                print(f"Agent {self.agent_id}: Initializing SI trackers for policy network")
                self.policy_si = SILoss(
                    model=self.policy,
                    importance=self.si_importance,
                    damping=self.si_damping,
                    exclude_layers=self.excluded_policy_layers if hasattr(self, 'excluded_policy_layers') else [],
                    device=self.device
                )
            
            # Initialize seen true states if not already initialized
            if not hasattr(self, 'seen_true_states'):
                self.seen_true_states = set()
                self.state_belief_si_trackers = {}
                self.state_policy_si_trackers = {}
                
            # Reset counters for SI
            self.belief_si_step_count = 0
            self.policy_si_step_count = 0
            self.accumulated_belief_loss = 0
            self.accumulated_policy_loss = 0
            self.path_integrals_calculated = False
    
    def infer_latent(self, signal, neighbor_actions, reward, next_signal):
        """Infer latent state of neighbors based on our observations which already contain neighbor actions."""

        # Convert reward to tensor
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(self.device).squeeze(1)

        if self.use_gnn:
            # Use the GNN for inference
            mean, logvar, opponent_belief_distribution = self.inference_module(
                signal,
                neighbor_actions,
                reward_tensor,
                next_signal,
                self.current_latent
            )
        else:
            # Use the traditional encoder
            mean, logvar, opponent_belief_distribution = self.encoder(
                signal,
                neighbor_actions,
                reward_tensor,
                next_signal,
                self.current_latent
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
    
    def select_action(self):
        """Select action based on current belief and latent."""

        # Calculate fresh action logits for action selection
        action_logits = self.policy(self.current_belief, self.current_latent)

        # Store a detached copy for caching
        self.action_logits = action_logits.detach()

        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)

        # Store probability of incorrect action for learning rate calculation
        self.action_probs_history.append(action_probs.squeeze(0).detach().cpu().numpy())
        
        # Sample action from distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        
        # Alternatively, use argmax for deterministic action selection
        #action = action_probs.argmax(dim=-1).item()

        
        return action, action_probs.squeeze(0).detach().cpu().numpy()
    
    def train(self, batch_size=32, sequence_length=32):
        """Train the agent using sequential data from the replay buffer."""
        # Sample sequential data from the replay buffer
        batch_sequences = self.replay_buffer.sample(batch_size, sequence_length, mode="sequence")
        
        # Update networks using sequential data
        return self.update(batch_sequences)
    
    def update(self, batch_sequences):
        """
        Update parameters based on a batch of sequences.
        
        Args:
            batch_sequences: Batch of environment interactions from replay buffer
        """
        # Unpack sequences
        signals, neighbor_actions, beliefs, latents, actions, rewards, next_signals, next_beliefs, next_latents, means, logvars = batch_sequences
        
        # Update inference module
        inference_loss = self._update_inference(signals, neighbor_actions, next_signals, next_latents, means, logvars)
        
        # Update critics
        critic_loss = self._update_critics(signals, neighbor_actions, beliefs, latents, actions, neighbor_actions, rewards, next_signals, next_beliefs, next_latents)
        
        # Update policy
        policy_loss = self._update_policy(beliefs, latents, actions, neighbor_actions)
        
        # Update Transformer for adaptive belief
        transformer_loss = self._update_transformer(signals, neighbor_actions, beliefs, next_signals)
        
        # Update target networks
        self._update_targets()
        
        return {
            'inference_loss': inference_loss,
            'critic_loss': critic_loss,
            'policy_loss': policy_loss,
            'transformer_loss': transformer_loss
        }
    
    def _compute_belief_loss(self, signals, neighbor_actions, beliefs, next_signals):
        """Compute the base belief loss (without SI component)."""
        # Forward pass
        _, belief_distributions = self.belief_processor(signals, neighbor_actions, beliefs)
        
        # Calculate belief loss (cross-entropy)
        belief_loss = F.binary_cross_entropy(belief_distributions, next_signals, reduction='mean')
        
        return belief_loss
    
    def _compute_policy_loss(self, beliefs, latents, actions):
        """Compute the base policy loss."""
        # Debug print
        print(f"Agent {self.agent_id}: _compute_policy_loss shapes:")
        print(f"  beliefs shape: {beliefs.shape}")
        print(f"  latents shape: {latents.shape}")
        print(f"  actions shape: {actions.shape}")
        
        # Get action logits
        action_logits = self.policy(beliefs, latents)
        print(f"  action_logits shape: {action_logits.shape}")
        
        # Ensure actions has the right shape for cross_entropy
        # cross_entropy expects target of shape [batch_size] for inputs of shape [batch_size, num_classes]
        if action_logits.dim() == 3 and actions.dim() == 1:
            # If action_logits has shape [batch_size, 2, 2] and actions has shape [batch_size],
            # reshape action_logits to [batch_size, 2]
            if action_logits.shape[1] == 2 and action_logits.shape[2] == 2:
                action_logits = action_logits[:, 0, :]
                print(f"  Reshaped action_logits shape: {action_logits.shape}")
                
        # For cross_entropy, if action_logits is [batch_size, num_classes] then actions should be [batch_size]
        # If actions is [batch_size, 1], we need to squeeze it
        if actions.dim() == 2 and actions.shape[1] == 1:
            actions = actions.squeeze(1)
            print(f"  Squeezed actions shape: {actions.shape}")
            
        # If action_logits has shape [batch_size, 1, num_classes], reshape it to [batch_size, num_classes]
        if action_logits.dim() == 3 and action_logits.shape[1] == 1:
            action_logits = action_logits.squeeze(1)
            print(f"  Squeezed action_logits shape: {action_logits.shape}")
            
        # Ensure action indices are correct - they should be in the range [0, num_classes-1]
        if actions.max() >= action_logits.shape[1]:
            # Scale actions to be within the valid range
            actions = torch.clamp(actions, 0, action_logits.shape[1] - 1)
            print(f"  Clamped actions range: [{actions.min()}, {actions.max()}]")
            
        # Compute cross entropy loss
        policy_loss = F.cross_entropy(action_logits, actions)
        
        return policy_loss
    
    def _update_inference(self, signals, neighbor_actions, next_signals, next_latents, means, logvars):
        """Update inference module with FURTHER-style temporal KL."""
        
        # Add debug prints to track tensor shapes
        print(f"Agent {self.agent_id}: _update_inference called")
        print(f"  signals shape: {signals.shape}")
        print(f"  neighbor_actions shape: {neighbor_actions.shape}")
        print(f"  next_signals shape: {next_signals.shape}")
        print(f"  next_latents shape: {next_latents.shape}")
        if means is not None:
            print(f"  means shape: {means.shape}")
            print(f"  logvars shape: {logvars.shape}")
        
        if self.use_gnn:
            # For GNN inference module
            # We need dummy rewards for the GNN forward pass
            batch_size = signals.shape[0]
            dummy_rewards = torch.zeros(batch_size, device=self.device)
            
            # Forward GNN to get z_t
            current_z, _ = self.inference_module(signals, neighbor_actions, dummy_rewards)
            
            # Forward GNN again to get z_{t+1}
            # Since GNN parameters are shared for both current and next time step
            next_z, _ = self.inference_module(next_signals, torch.zeros_like(neighbor_actions), dummy_rewards)
            
            # Loss for tracking next latent
            reconstruct_loss = F.mse_loss(next_z, next_latents)
            kl_div = torch.tensor(0.0, device=self.device)
        else:
            # For encoder-decoder module
            # Forward encoder with current observation
            current_z, mu, logvar = self.encoder(signals, neighbor_actions)
            
            # Only compute KL loss if means and logvars provided from replay buffer
            if means is not None and logvars is not None:
                # Compute KL divergence from temporal consistency: p(z_{t+1}|z_t, a_t) || q(z_{t+1}|o_{t+1})
                prior_mu = means
                prior_logvar = logvars
                
                # Compute KL divergence
                kl_div = -0.5 * torch.sum(1 + logvar - prior_logvar - 
                                         (torch.exp(logvar) + (mu - prior_mu).pow(2)) / 
                                          torch.exp(prior_logvar), dim=1)
                kl_div = kl_div.mean()
            else:
                kl_div = torch.tensor(0.0, device=self.device)
                
            # Forward encoder with next observation
            next_z_recon, _, _ = self.encoder(next_signals, torch.zeros_like(neighbor_actions))
            
            # Loss for tracking next latent
            reconstruct_loss = F.mse_loss(next_z_recon, next_latents)
        
        # Total loss
        inference_loss = reconstruct_loss + self.kl_weight * kl_div
        
        # Update networks with SI regularization
        self.encoder_optimizer.zero_grad()
        
        # We need to retain the graph if we're using SI, so we can compute parameter importance later
        # This is important when the same graph is used for multiple backward passes
        retain_graph = self.use_si
        
        inference_loss.backward(retain_graph=retain_graph)
        
        torch.nn.utils.clip_grad_norm_(
            self.encoder.parameters(),
            max_norm=1.0
        )
        
        self.encoder_optimizer.step()
        
        # Update SI trackers for encoder if using SI
        if self.use_si and hasattr(self, 'inference_si'):
            self.inference_si.update_tracker(inference_loss.item())
            
        return inference_loss.item()
    
    def _calculate_temporal_kl_divergence(self, means_seq, logvars_seq):
        """Calculate KL divergence between sequential latent states (temporal smoothing)."""

        # KL(N(mu,E), N(m, S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m - mu)^T S^-1 (m - mu)))
        # Ref: https://github.com/lmzintgraf/varibad/blob/master/vae.py
        # Ref: https://github.com/dkkim93/further/blob/main/algorithm/further/agent.py

        kl_first_term = torch.sum(logvars_seq[:-1, :], dim=-1) - torch.sum(logvars_seq[1:, :], dim=-1)
        kl_second_term = self.latent_dim
        kl_third_term = torch.sum(1. / torch.exp(logvars_seq[:-1, :]) * torch.exp(logvars_seq[1:, :]), dim=-1)
        kl_fourth_term = (means_seq[:-1, :] - means_seq[1:, :]) / torch.exp(logvars_seq[:-1, :]) * (means_seq[:-1, :] - means_seq[1:, :])
        kl_fourth_term = kl_fourth_term.sum(dim=-1)
        
        kl = 0.5 * (kl_first_term - kl_second_term + kl_third_term + kl_fourth_term)

        return self.kl_weight * torch.mean(kl)
    
    def _update_critics(self, signals, neighbor_actions, beliefs, latents, actions, next_neighbor_actions, rewards, next_signals, next_beliefs, next_latents):
        """Update Q-networks."""
        # Get current Q-values
        q1 = self.q_network1(beliefs, latents, neighbor_actions).gather(1, actions.unsqueeze(1))
        q2 = self.q_network2(beliefs, latents, neighbor_actions).gather(1, actions.unsqueeze(1))
        
        # Debug shapes
        print(f"Agent {self.agent_id}: critic q shapes:")
        print(f"  q1 shape: {q1.shape}")
        
        # Compute next action probabilities
        with torch.no_grad():
            # Calculate fresh action logits for critic update
            next_action_logits = self.policy(next_beliefs, next_latents)
            next_action_probs = F.softmax(next_action_logits, dim=1)
            next_log_probs = F.log_softmax(next_action_logits, dim=1)
            entropy = -torch.sum(next_action_probs * next_log_probs, dim=1, keepdim=True)
            
            # Compute Q-values with predicted next neighbor actions
            next_q1 = self.target_q_network1(next_beliefs, next_latents, next_neighbor_actions)
            next_q2 = self.target_q_network2(next_beliefs, next_latents, next_neighbor_actions)
            
            # Take minimum
            next_q = torch.min(next_q1, next_q2)
            
            # Debug print tensor shapes
            print(f"Agent {self.agent_id}: _update_critics tensor shapes:")
            print(f"  next_action_probs shape: {next_action_probs.shape}")
            print(f"  next_q shape: {next_q.shape}")
            print(f"  rewards shape: {rewards.shape}")
            
            # Make sure tensor dimensions match before computing expected_q
            # Handle 3D next_action_probs
            if next_action_probs.dim() == 3:
                # If it has shape [batch_size, 2, 2], squeeze or reshape
                if next_action_probs.shape[1] == 2 and next_action_probs.shape[2] == 2:
                    # Take the first dimension as the batch, and the second dimension as the action dim
                    next_action_probs = next_action_probs[:, 0, :]
                    print(f"  Reshaped next_action_probs shape: {next_action_probs.shape}")
            
            # Handle if dimensions still don't match
            if next_action_probs.shape[1] != next_q.shape[1]:
                # If next_q has shape [batch_size, 1, action_dim], squeeze the middle dimension
                if next_q.dim() == 3:
                    next_q = next_q.squeeze(1)
                
                # If shapes still don't match, try to expand next_q to match next_action_probs
                if next_action_probs.shape[1] != next_q.shape[1]:
                    batch_size = next_action_probs.shape[0]
                    action_dim = next_action_probs.shape[1]
                    
                    # Create a new tensor of the right shape, filled with zeros
                    expanded_next_q = torch.zeros((batch_size, action_dim), device=self.device)
                    
                    # Fill with values from next_q where possible
                    for i in range(min(action_dim, next_q.shape[1])):
                        expanded_next_q[:, i] = next_q[:, i]
                    
                    next_q = expanded_next_q
                    print(f"  Expanded next_q shape: {next_q.shape}")
            
            # Expected Q-value
            expected_q = (next_action_probs * next_q).sum(dim=1, keepdim=True)
            print(f"  expected_q shape: {expected_q.shape}")
            
            # Make sure rewards has the right shape
            if rewards.shape != expected_q.shape:
                # If rewards has shape [batch_size] and expected_q has shape [batch_size, 1], add a dimension
                if rewards.dim() == 1 and expected_q.dim() == 2:
                    rewards = rewards.unsqueeze(1)
                    print(f"  Reshaped rewards shape: {rewards.shape}")
                
                # If there's still a mismatch, try to reshape rewards to match expected_q
                if rewards.shape != expected_q.shape:
                    batch_size = expected_q.shape[0]
                    # Create a new tensor of the right shape
                    reshaped_rewards = torch.zeros_like(expected_q)
                    # Fill with values from rewards where possible
                    for i in range(min(batch_size, rewards.shape[0])):
                        if rewards.dim() == 1:
                            reshaped_rewards[i, 0] = rewards[i]
                        else:
                            reshaped_rewards[i, 0] = rewards[i, 0]
                    rewards = reshaped_rewards
                    print(f"  Fully reshaped rewards shape: {rewards.shape}")
            
            # Add entropy
            expected_q = expected_q + self.entropy_weight * entropy
            
            # Compute target
            if self.discount_factor > 0:  # Discounted return
                target_q = rewards + self.discount_factor * expected_q
            else:  # Average reward
                target_q = rewards - self.gain_parameter + expected_q
            
            # Ensure target_q has the same shape as q1 and q2
            if target_q.shape != q1.shape:
                print(f"  target_q shape before reshape: {target_q.shape}, q1 shape: {q1.shape}")
                
                # Simpler approach: just create a new tensor with the correct shape and copy values
                batch_size = q1.shape[0]
                reshaped_target_q = torch.zeros_like(q1)
                
                # Handle different target_q dimensions
                if target_q.dim() == 2 and q1.dim() == 2:
                    # Both are 2D, just copy the values directly if shapes are compatible
                    # E.g., target_q: [8, 1], q1: [8, 1]
                    if target_q.shape[0] == q1.shape[0]:
                        for i in range(batch_size):
                            reshaped_target_q[i, 0] = target_q[i, 0]
                
                elif target_q.dim() == 3 and q1.dim() == 2:
                    # target_q is 3D, q1 is 2D
                    # E.g., target_q: [8, 8, 2], q1: [8, 1]
                    # Just take the first element from each batch
                    for i in range(min(batch_size, target_q.shape[0])):
                        reshaped_target_q[i, 0] = target_q[i, 0, 0]
                
                # If all else fails, just use the original q1 values (this is a fallback)
                if torch.all(reshaped_target_q == 0):
                    print("  Failed to reshape target_q, using original q1 values")
                    reshaped_target_q = q1.detach()
                
                target_q = reshaped_target_q
                print(f"  Reshaped target_q shape: {target_q.shape}")
        
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
            max_norm=1.0
        )
        self.q_optimizer.step()
        
        if self.discount_factor == 0:
            self.gain_optimizer.step()
        
        return q_loss.item()
    
    def _update_policy(self, beliefs, latents, actions, neighbor_actions):
        """Update policy network."""
        print(f"Agent {self.agent_id}: Updating policy network (beliefs shape: {beliefs.shape})")
        
        # Zero gradients
        self.policy_optimizer.zero_grad()
        
        # Compute the base policy loss
        policy_loss = self._compute_policy_loss(beliefs, latents, actions)
        
        # Calculate SI loss for policy if using SI
        si_policy_loss = torch.tensor(0.0, device=self.device)
        if self.use_si and hasattr(self, 'policy_si'):
            si_policy_loss = self.policy_si.calculate_loss()
            
        # Combine losses
        combined_loss = policy_loss + si_policy_loss
        
        # We need to retain the graph if we're using SI, so we can compute parameter importance later
        retain_graph = self.use_si
        
        # Backward pass
        combined_loss.backward(retain_graph=retain_graph)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            max_norm=1.0
        )
        
        # Update
        self.policy_optimizer.step()
        
        # Update SI trackers for policy if using SI
        if self.use_si and hasattr(self, 'policy_si'):
            print(f"Agent {self.agent_id}: Updated policy SI tracker (step {self.policy_si.step_count}, loss: {policy_loss.item():.6f})")
            self.policy_si.update_tracker(policy_loss.item())
            
        return policy_loss.item()
    
    def _update_transformer(self, signals, neighbor_actions, beliefs, next_signals):
        """Update belief processor using transformer model."""
        # Zero gradients
        self.transformer_optimizer.zero_grad()
        
        # Compute the base belief loss
        belief_loss = self._compute_belief_loss(signals, neighbor_actions, beliefs, next_signals)
        
        # Calculate SI loss for belief processor if using SI
        si_belief_loss = torch.tensor(0.0, device=self.device)
        if self.use_si and hasattr(self, 'belief_si'):
            si_belief_loss = self.belief_si.calculate_loss()
            
        # Combine losses
        combined_loss = belief_loss + si_belief_loss
        
        # We need to retain the graph if we're using SI, so we can compute parameter importance later
        retain_graph = self.use_si
        
        # Backward pass
        combined_loss.backward(retain_graph=retain_graph)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.belief_processor.parameters(),
            max_norm=1.0
        )
        
        # Update
        self.transformer_optimizer.step()
        
        # Update SI trackers for belief processor if using SI
        if self.use_si and hasattr(self, 'belief_si'):
            self.belief_si.update_tracker(belief_loss.item())
            
        return belief_loss.item()
    
    def _update_targets(self):
        """Update target networks."""
        for target_param, param in zip(self.target_q_network1.parameters(), self.q_network1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_rate) + 
                param.data * self.target_update_rate
            )
        
        for target_param, param in zip(self.target_q_network2.parameters(), self.q_network2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_rate) + 
                param.data * self.target_update_rate
            )
    
    def save(self, path):
        """Save the agent's networks to a file."""
        checkpoint = {
            'belief_processor': self.belief_processor.state_dict(),
            'policy': self.policy.state_dict(),
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'target_q_network1': self.target_q_network1.state_dict(),
            'target_q_network2': self.target_q_network2.state_dict(),
            'gain_parameter': self.gain_parameter,
            'use_gnn': self.use_gnn
        }
        
        # Save the appropriate inference module
        if self.use_gnn:
            checkpoint['inference_module'] = self.inference_module.state_dict()
        else:
            checkpoint['encoder'] = self.encoder.state_dict()
            checkpoint['decoder'] = self.decoder.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path, evaluation_mode=False):
        """
        Load agent model.
        
        Args:
            path: Path to the saved model
            evaluation_mode: If True, sets the model to evaluation mode after loading
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check if we're loading a GRU model into a Transformer model
        is_gru_to_transformer = False
        try:
            # Try to load belief processor (may fail if architecture changed from GRU to Transformer)
            self.belief_processor.load_state_dict(checkpoint['belief_processor'])
        except RuntimeError as e:
            print(f"Warning: Could not load belief processor due to architecture change: {e}")
            print("Using the new Transformer belief processor with initialized weights.")
            print("Attempting to transfer knowledge from GRU to Transformer...")
            is_gru_to_transformer = True
            
            # Try to transfer knowledge from GRU to Transformer
            self._transfer_gru_to_transformer_knowledge(checkpoint)
            
        # Load other components that should be compatible
        try:
            if self.use_gnn:
                self.inference_module.load_state_dict(checkpoint['inference_module'])
            else:
                self.encoder.load_state_dict(checkpoint['encoder'])
                self.decoder.load_state_dict(checkpoint['decoder'])
            self.policy.load_state_dict(checkpoint['policy'])
        except RuntimeError as e:
            print(f"Warning: Could not load some components: {e}")
            
        # Handle Q-networks
        try:
            self.q_network1.load_state_dict(checkpoint['q_network1'])
            self.q_network2.load_state_dict(checkpoint['q_network2'])
            self.target_q_network1.load_state_dict(checkpoint['target_q_network1'])
            self.target_q_network2.load_state_dict(checkpoint['target_q_network2'])
        except RuntimeError as e:
            print(f"Warning: Could not load Q-networks due to architecture changes: {e}")
            print("Initializing new Q-networks. You may need to retrain the model.")
        
        try:
            self.gain_parameter.data = checkpoint['gain_parameter']
        except:
            print("Warning: Could not load gain parameter.")
        
        # Reset internal state after loading
        self.reset_internal_state()
        
        # Set the model to evaluation mode if requested
        if evaluation_mode:
            self.set_eval_mode()
            print(f"Model set to evaluation mode.")
        else:
            self.set_train_mode()
            print(f"Model set to training mode.")
        
        # If we transferred from GRU to Transformer, recommend retraining
        if is_gru_to_transformer:
            print("Knowledge transfer from GRU to Transformer attempted.")
            print("For best performance, you should retrain the model for a few episodes.")
            
    def _transfer_gru_to_transformer_knowledge(self, checkpoint):
        """
        Transfer knowledge from a GRU model to a Transformer model.
        This helps preserve some of the learned knowledge when switching architectures.
        """
        try:
            # The most important part to transfer is the belief head weights
            # which map from hidden state to belief distribution
            if 'belief_processor' in checkpoint:
                gru_state_dict = checkpoint['belief_processor']
                
                # Transfer belief head weights if they have the same dimensions
                if 'belief_head.weight' in gru_state_dict and gru_state_dict['belief_head.weight'].size() == self.belief_processor.belief_head.weight.size():
                    self.belief_processor.belief_head.weight.data.copy_(gru_state_dict['belief_head.weight'])
                    self.belief_processor.belief_head.bias.data.copy_(gru_state_dict['belief_head.bias'])
                    print("Successfully transferred belief head weights from GRU to Transformer.")
                    
                # We can also try to initialize the input projection with GRU input weights
                if 'gru.weight_ih_l0' in gru_state_dict:
                    # The input weights of GRU can be used to initialize part of the input projection
                    gru_input_weights = gru_state_dict['gru.weight_ih_l0']
                    input_dim = min(gru_input_weights.size(1), self.belief_processor.input_projection.weight.size(1))
                    output_dim = min(gru_input_weights.size(0) // 3, self.belief_processor.input_projection.weight.size(0))
                    
                    # Copy the reset gate weights (first third of GRU weights)
                    self.belief_processor.input_projection.weight.data[:output_dim, :input_dim].copy_(
                        gru_input_weights[:output_dim, :input_dim]
                    )
                    print("Partially initialized Transformer input projection with GRU weights.")
                    
        except Exception as e:
            print(f"Error during knowledge transfer: {e}")
            print("Continuing with randomly initialized Transformer.")

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
        return self.current_opponent_belief_distribution if hasattr(self, 'current_opponent_belief_distribution') else None
        
    def calculate_path_integrals(self, replay_buffer):
        """
        Register path integrals accumulated during training.
        
        This method is called when a new task is detected.
        Instead of recalculating path integrals from the replay buffer,
        it uses the accumulated path integrals from the backward passes
        during the regular training.
        
        Args:
            replay_buffer: Optional replay buffer (not used in this implementation)
        """
        if not self.use_si:
            return
            
        if not hasattr(self, 'belief_si') or not hasattr(self, 'policy_si'):
            print(f"Agent {self.agent_id}: SI trackers not initialized. Skipping path integral registration.")
            return
            
        print(f"Agent {self.agent_id}: Registering accumulated path integrals")
        print(f"Belief network updates: {self.belief_si_step_count}, avg loss: {self.accumulated_belief_loss/max(1, self.belief_si_step_count):.6f}")
        print(f"Policy network updates: {self.policy_si_step_count}, avg loss: {self.accumulated_policy_loss/max(1, self.policy_si_step_count):.6f}")
        
        # Calculate average parameter change magnitudes
        belief_param_change_magnitude = 0.0
        for name, param in self.belief_processor.named_parameters():
            if param.requires_grad and name in self.belief_si.previous_params:
                param_diff = torch.sum(torch.abs(param.data - self.belief_si.previous_params[name])).item()
                belief_param_change_magnitude += param_diff
        
        policy_param_change_magnitude = 0.0
        for name, param in self.policy.named_parameters():
            if param.requires_grad and name in self.policy_si.previous_params:
                param_diff = torch.sum(torch.abs(param.data - self.policy_si.previous_params[name])).item()
                policy_param_change_magnitude += param_diff
                
        print(f"Parameter change magnitudes - Belief: {belief_param_change_magnitude:.6f}, Policy: {policy_param_change_magnitude:.6f}")
        
        # Apply dynamic importance scaling based on parameter change magnitude
        current_true_state = self.current_true_state
        print(f"Current true state: {current_true_state}")
        
        # Get the base importance value
        base_importance = self.si_importance
        
        # If we've seen other tasks before, adjust the importance scaling
        if len(self.seen_true_states) > 0:
            # Calculate scaling factor inversely proportional to parameter change magnitude
            if hasattr(self, 'reference_belief_change') and self.reference_belief_change > 0:
                belief_scale = self.reference_belief_change / max(belief_param_change_magnitude, 1e-10)
                belief_scale = min(max(belief_scale, 0.1), 10.0)
            else:
                self.reference_belief_change = belief_param_change_magnitude
                belief_scale = 1.0
                
            if hasattr(self, 'reference_policy_change') and self.reference_policy_change > 0:
                policy_scale = self.reference_policy_change / max(policy_param_change_magnitude, 1e-10)
                policy_scale = min(max(policy_scale, 0.1), 10.0)
            else:
                self.reference_policy_change = policy_param_change_magnitude
                policy_scale = 1.0
                
            # Apply scaling to importance
            belief_importance = base_importance * belief_scale
            policy_importance = base_importance * policy_scale
            
            print(f"Applying dynamic importance scaling - Belief: {belief_scale:.4f} (adjusted importance: {belief_importance:.4f}), "
                  f"Policy: {policy_scale:.4f} (adjusted importance: {policy_importance:.4f})")
            
            # Update the tracker importance values
            self.belief_si.importance = belief_importance
            self.policy_si.importance = policy_importance
        else:
            # First task encountered - save reference values
            self.reference_belief_change = belief_param_change_magnitude
            self.reference_policy_change = policy_param_change_magnitude
            print(f"First task encountered, setting reference parameter change values")
        
        # Register tasks with SI (this uses the accumulated path integrals in the trackers)
        self.belief_si.register_task()
        self.policy_si.register_task()
        
        # Store task-specific trackers for visualization and comparison across tasks
        if hasattr(self, 'current_true_state') and self.current_true_state is not None:
            print(f"Storing SI trackers for true state {self.current_true_state}")
            
            # Create copies of the trackers so they don't get modified as we continue training
            self.state_belief_si_trackers[self.current_true_state] = self._clone_si_tracker(self.belief_si)
            self.state_policy_si_trackers[self.current_true_state] = self._clone_si_tracker(self.policy_si)
        
        # Verify the importance scores after registration
        sum_belief_importance = sum(torch.sum(torch.abs(imp)).item() 
                                   for name, imp in self.belief_si.importance_scores.items())
        sum_policy_importance = sum(torch.sum(torch.abs(imp)).item() 
                                   for name, imp in self.policy_si.importance_scores.items())
        
        print(f"Sum of belief importance scores after registration: {sum_belief_importance:.8f}")
        print(f"Sum of policy importance scores after registration: {sum_policy_importance:.8f}")
        
        # Reset counters for next task
        self.belief_si_step_count = 0
        self.policy_si_step_count = 0
        self.accumulated_belief_loss = 0
        self.accumulated_policy_loss = 0
        
        self.path_integrals_calculated = True
    
    def _clone_si_tracker(self, tracker):
        """Create a deep copy of an SI tracker for state comparison."""
        # Create a new tracker with the same parameters
        cloned_tracker = SILoss(
            model=tracker.model,
            importance=tracker.importance,
            damping=tracker.damping,
            exclude_layers=tracker.exclude_layers,
            device=tracker.device
        )
        
        # Copy importance scores (these need to be deep copied)
        for name, importance in tracker.importance_scores.items():
            cloned_tracker.importance_scores[name] = importance.clone().detach()
        
        # Copy previous parameters (reference parameters)
        if hasattr(tracker, 'previous_params'):
            for name, param in tracker.previous_params.items():
                cloned_tracker.previous_params[name] = param.clone().detach()
                
        # Copy path integrals if they exist
        if hasattr(tracker, 'param_path_integrals'):
            for name, integral in tracker.param_path_integrals.items():
                cloned_tracker.param_path_integrals[name] = integral.clone().detach()
                
        return cloned_tracker
    
    def end_episode(self):
        """
        Backward compatibility method - does nothing in the continuous version.
        The internal state is maintained across what would have been episode boundaries.
        """
        pass
    