import torch
import numpy as np
from collections import deque
from modules.utils import get_best_device

class ReplayBuffer:
    """ReplayBuffer for storing agent experiences with batched tensor storage."""
    
    def __init__(self, capacity, observation_dim, belief_dim, latent_dim, device, sequence_length=32):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            observation_dim: Dimension of observations
            belief_dim: Dimension of belief states
            latent_dim: Dimension of latent states
            device: Device to store tensors on
            sequence_length: Length of sequences for RNN-based agents
        """
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.belief_dim = belief_dim
        self.latent_dim = latent_dim
        self.device = device
        self.sequence_length = sequence_length
        
        # Buffer for transitions
        self.buffer = []
        self.position = 0
        
        # Tensor storage for efficient batching
        self.signals = None
        self.neighbor_actions = None
        self.beliefs = None 
        self.latents = None
        self.actions = None
        self.rewards = None
        self.next_signals = None
        self.next_beliefs = None
        self.next_latents = None
        self.means = None
        self.logvars = None
        
    def push(self, signal, neighbor_actions, belief, latent, action, reward, 
             next_signal, next_belief, next_latent, mean=None, logvar=None):
        """
        Store a transition in the buffer.
        
        Args:
            signal: Current observation
            neighbor_actions: Actions of neighbors
            belief: Current belief state
            latent: Current latent state
            action: Action taken
            reward: Reward received
            next_signal: Next observation
            next_belief: Next belief state
            next_latent: Next latent state
            mean: Mean of latent distribution (for VAE)
            logvar: Log variance of latent distribution (for VAE)
        """
        transition = (signal, neighbor_actions, belief, latent, action, reward, 
                     next_signal, next_belief, next_latent, mean, logvar)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        
        # If we have filled the buffer at least once, convert to tensors
        if len(self.buffer) == self.capacity:
            self._update_tensors()
    
    def _update_tensors(self):
        """Convert stored transitions to tensors for efficient batching."""
        # Extract each component and stack into tensors
        signals, neighbor_actions, beliefs, latents, actions, rewards, next_signals, next_beliefs, next_latents, means, logvars = zip(*self.buffer)
        
        # Standardize dimensions before stacking to ensure consistent shapes
        # For latents, ensure all have same dimension [batch, latent_dim]
        standardized_latents = []
        standardized_next_latents = []
        
        for lat in latents:
            if lat.dim() == 3:  # Shape [1, 1, latent_dim]
                standardized_latents.append(lat.squeeze(1))  # Convert to [1, latent_dim]
            else:
                standardized_latents.append(lat)
                
        for next_lat in next_latents:
            if next_lat.dim() == 3:  # Shape [1, 1, latent_dim]
                standardized_next_latents.append(next_lat.squeeze(1))  # Convert to [1, latent_dim]
            else:
                standardized_next_latents.append(next_lat)
        
        self.signals = torch.stack(signals).to(self.device)
        self.neighbor_actions = torch.stack(neighbor_actions).to(self.device)
        self.beliefs = torch.stack([self._standardize_belief_state(b) for b in beliefs]).to(self.device)
        self.latents = torch.stack(standardized_latents).to(self.device)
        self.actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        self.rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        self.next_signals = torch.stack(next_signals).to(self.device)
        self.next_beliefs = torch.stack([self._standardize_belief_state(b) for b in next_beliefs]).to(self.device)
        self.next_latents = torch.stack(standardized_next_latents).to(self.device)
        
        # Handle optional VAE parameters
        if means[0] is not None:
            standardized_means = []
            for m in means:
                if m.dim() == 3:  # Shape [1, 1, latent_dim]
                    standardized_means.append(m.squeeze(1))  # Convert to [1, latent_dim]
                else:
                    standardized_means.append(m)
            self.means = torch.stack(standardized_means).to(self.device)
            
        if logvars[0] is not None:
            standardized_logvars = []
            for lv in logvars:
                if lv.dim() == 3:  # Shape [1, 1, latent_dim]
                    standardized_logvars.append(lv.squeeze(1))  # Convert to [1, latent_dim]
                else:
                    standardized_logvars.append(lv)
            self.logvars = torch.stack(standardized_logvars).to(self.device)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def sample(self, batch_size=32, sequence_length=None):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            sequence_length: Length of sequences to sample (for rnn-based agents)
            
        Returns:
            Tuple of (signals, neighbor_actions, beliefs, latents, actions, rewards, 
                      next_signals, next_beliefs, next_latents, means, logvars)
        """
        # Print debug info
        print(f"ReplayBuffer: Attempting to sample batch_size={batch_size}, buffer size={len(self)}")
        
        # Check if buffer has enough samples
        if len(self) < batch_size:
            print(f"ReplayBuffer: Not enough samples in buffer (need {batch_size}, have {len(self)})")
            return None
        
        # If tensors aren't initialized yet, use the old approach
        if self.signals is None:
            print(f"ReplayBuffer: Tensors not initialized, sampling directly from buffer")
            # Sample indices
            indices = np.random.randint(0, len(self.buffer), batch_size)
            batch = [self.buffer[i] for i in indices]
            
            # Unzip the batch
            signals, neighbor_actions, beliefs, latents, actions, rewards, next_signals, next_beliefs, next_latents, means, logvars = zip(*batch)
            
            # Convert to tensors with standardized dimensions
            signals = torch.stack(signals).to(self.device)
            neighbor_actions = torch.stack(neighbor_actions).to(self.device)
            beliefs = torch.stack([self._standardize_belief_state(b) for b in beliefs]).to(self.device)
            
            # Standardize latent state dimensions
            standardized_latents = []
            for lat in latents:
                if lat.dim() == 3:  # Shape [1, 1, latent_dim]
                    standardized_latents.append(lat.squeeze(1))  # Convert to [1, latent_dim]
                else:
                    standardized_latents.append(lat)
            latents = torch.stack(standardized_latents).to(self.device)
            
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
            next_signals = torch.stack(next_signals).to(self.device)
            next_beliefs = torch.stack([self._standardize_belief_state(b) for b in next_beliefs]).to(self.device)
            
            # Standardize next latent state dimensions
            standardized_next_latents = []
            for next_lat in next_latents:
                if next_lat.dim() == 3:  # Shape [1, 1, latent_dim]
                    standardized_next_latents.append(next_lat.squeeze(1))  # Convert to [1, latent_dim]
                else:
                    standardized_next_latents.append(next_lat)
            next_latents = torch.stack(standardized_next_latents).to(self.device)
            
            # Handle optional VAE parameters
            if means[0] is not None:
                standardized_means = []
                for m in means:
                    if m.dim() == 3:  # Shape [1, 1, latent_dim]
                        standardized_means.append(m.squeeze(1))  # Convert to [1, latent_dim]
                    else:
                        standardized_means.append(m)
                means = torch.stack(standardized_means).to(self.device)
            else:
                means = None
            
            if logvars[0] is not None:
                standardized_logvars = []
                for lv in logvars:
                    if lv.dim() == 3:  # Shape [1, 1, latent_dim]
                        standardized_logvars.append(lv.squeeze(1))  # Convert to [1, latent_dim]
                    else:
                        standardized_logvars.append(lv)
                logvars = torch.stack(standardized_logvars).to(self.device)
            else:
                logvars = None
                
            print(f"ReplayBuffer: Successfully sampled batch of size {batch_size}")
            
            return (
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
                logvars
            )
        
        # Get random indices
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        
        # Get batches for each tensor
        signals_batch = self.signals[indices]
        neighbor_actions_batch = self.neighbor_actions[indices]
        beliefs_batch = self.beliefs[indices]
        latents_batch = self.latents[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        next_signals_batch = self.next_signals[indices]
        next_beliefs_batch = self.next_beliefs[indices]
        next_latents_batch = self.next_latents[indices]
        means_batch = self.means[indices] if self.means is not None else None
        logvars_batch = self.logvars[indices] if self.logvars is not None else None
        
        print(f"ReplayBuffer: Successfully sampled batch of size {batch_size}")
        
        return (
            signals_batch, 
            neighbor_actions_batch, 
            beliefs_batch, 
            latents_batch, 
            actions_batch, 
            rewards_batch, 
            next_signals_batch, 
            next_beliefs_batch, 
            next_latents_batch,
            means_batch,
            logvars_batch
        )
    
    def _standardize_belief_state(self, belief):
        """
        Standardize belief state shape for consistent processing.
        
        Args:
            belief: Belief state tensor
            
        Returns:
            Standardized belief state tensor with shape [1, belief_dim]
        """
        # Handle various input shapes gracefully
        if belief.dim() == 3:
            # Shape [1, 1, belief_dim]
            return belief.squeeze(1)
        elif belief.dim() == 2 and belief.size(0) == 1:
            # Shape [1, belief_dim]
            return belief
        elif belief.dim() == 1:
            # Shape [belief_dim]
            return belief.unsqueeze(0)
        else:
            # Fall back to reshaping
            return belief.view(1, -1)
    
    def end_trajectory(self):
        """Backward compatibility method - does nothing in the continuous version."""
        pass
    
    def _process_transitions(self, transitions):
        """Process a list of transitions into batched tensors."""
        if not transitions:
            return None
            
        # Unpack transitions
        signals, neighbor_actions, beliefs, latents, actions, rewards, \
        next_signals, next_beliefs, next_latents, means, logvars = zip(*transitions)
        
        # For tensors that are already torch tensors, we need to detach them
        signals_list = []
        for s in signals:
            if isinstance(s, torch.Tensor):
                signals_list.append(s.detach())
            elif isinstance(s, (int, float)):
                signals_list.append(torch.tensor([s], dtype=torch.float32))
            else:
                signals_list.append(s)
        
        neighbor_actions_list = []
        for na in neighbor_actions:
            if isinstance(na, torch.Tensor):
                neighbor_actions_list.append(na.detach())
            elif isinstance(na, (int, float)):
                neighbor_actions_list.append(torch.tensor([na], dtype=torch.float32))
            else:
                neighbor_actions_list.append(na)
        
        # Process belief states to ensure consistent shape [1, batch_size, hidden_dim]
        beliefs_list = []
        next_beliefs_list = []
        for b, nb in zip(beliefs, next_beliefs):
            # Handle current belief
            if isinstance(b, torch.Tensor):
                b = self._standardize_belief_state(b.detach())
                beliefs_list.append(b)
            else:
                beliefs_list.append(torch.zeros(1, 1, self.belief_dim, device=self.device))
            
            # Handle next belief
            if isinstance(nb, torch.Tensor):
                nb = self._standardize_belief_state(nb.detach())
                next_beliefs_list.append(nb)
            else:
                next_beliefs_list.append(torch.zeros(1, 1, self.belief_dim, device=self.device))
        
        # Process latent states to ensure consistent shape [1, batch_size, latent_dim]
        latents_list = []
        next_latents_list = []
        for l, nl in zip(latents, next_latents):
            # Handle current latent
            if isinstance(l, torch.Tensor):
                if l.dim() == 1:  # [latent_dim]
                    l = l.unsqueeze(0)  # [1, latent_dim]
                if l.dim() == 2:  # [batch_size, latent_dim]
                    l = l.unsqueeze(0)  # [1, batch_size, latent_dim]
                latents_list.append(l.detach())
            else:
                # Create zero tensor if not a tensor
                latents_list.append(torch.zeros(1, 1, self.belief_dim, device=self.device))
            
            # Handle next latent
            if isinstance(nl, torch.Tensor):
                if nl.dim() == 1:  # [latent_dim]
                    nl = nl.unsqueeze(0)  # [1, latent_dim]
                if nl.dim() == 2:  # [batch_size, latent_dim]
                    nl = nl.unsqueeze(0)  # [1, batch_size, latent_dim]
                next_latents_list.append(nl.detach())
            else:
                # Create zero tensor if not a tensor
                next_latents_list.append(torch.zeros(1, 1, self.belief_dim, device=self.device))
        
        next_signals_list = []
        for ns in next_signals:
            if isinstance(ns, torch.Tensor):
                next_signals_list.append(ns.detach())
            elif isinstance(ns, (int, float)):
                next_signals_list.append(torch.tensor([ns], dtype=torch.float32))
            else:
                next_signals_list.append(ns)
        
        # Handle means and logvars which might be None or float for older entries
        means_list = []
        logvars_list = []
        for m, lv in zip(means, logvars):
            if m is not None and lv is not None:
                if isinstance(m, torch.Tensor) and isinstance(lv, torch.Tensor):
                    means_list.append(m.detach())
                    logvars_list.append(lv.detach())
                elif isinstance(m, (int, float)) and isinstance(lv, (int, float)):
                    means_list.append(torch.tensor([m], dtype=torch.float32))
                    logvars_list.append(torch.tensor([lv], dtype=torch.float32))
        
        try:
            # Stack tensors with consistent shapes
            signals = torch.stack(signals_list).to(self.device)
            neighbor_actions = torch.stack(neighbor_actions_list).to(self.device)
            beliefs = torch.cat([b.view(1, 1, -1) for b in beliefs_list], dim=1).to(self.device)  # Ensure consistent shape
            latents = torch.cat([l.view(1, 1, -1) for l in latents_list], dim=1).to(self.device)  # Ensure consistent shape
            
            # Convert actions to tensor if they're not already
            actions_tensor = []
            for a in actions:
                if isinstance(a, torch.Tensor):
                    actions_tensor.append(a.item())
                else:
                    actions_tensor.append(int(a))
            actions = torch.LongTensor(actions_tensor).to(self.device)
            
            # Convert rewards to tensor
            rewards_tensor = []
            for r in rewards:
                if isinstance(r, torch.Tensor):
                    rewards_tensor.append(r.item())
                else:
                    rewards_tensor.append(float(r))
            rewards = torch.FloatTensor(rewards_tensor).unsqueeze(1).to(self.device)
            
            next_signals = torch.stack(next_signals_list).to(self.device)
            next_beliefs = torch.cat([nb.view(1, 1, -1) for nb in next_beliefs_list], dim=1).to(self.device)  # Ensure consistent shape
            next_latents = torch.cat([nl.view(1, 1, -1) for nl in next_latents_list], dim=1).to(self.device)  # Ensure consistent shape
            
            # Only create means and logvars tensors if we have data
            means = torch.cat(means_list).to(self.device) if means_list else None
            logvars = torch.cat(logvars_list).to(self.device) if logvars_list else None
            
            return (signals, neighbor_actions, beliefs, latents, actions, rewards,
                    next_signals, next_beliefs, next_latents, means, logvars)
        except Exception as e:
            print(f"Error processing transitions: {e}")
            return None
    
    def _process_sequence_batch(self, sequences):
        """Process a batch of sequences for GRU training."""
        batch_data = []
        sequence_length = len(sequences[0])
        
        for t in range(sequence_length):
            # Get all transitions at time step t across all sequences
            time_step_transitions = [seq[t] for seq in sequences]
            
            # Process these transitions into batched tensors
            time_step_data = self._process_transitions(time_step_transitions)
            batch_data.append(time_step_data)
        
        return batch_data

    def get_sequential_latent_params(self):
        """Get all means and logvars in chronological order for temporal KL calculation."""
        if len(self.buffer) < 2:
            return None, None
            
        transitions = list(self.buffer)
        means = [t[9] for t in transitions if t[9] is not None]  # Means at index 9
        logvars = [t[10] for t in transitions if t[10] is not None]  # Logvars at index 10
        
        if not means or not logvars or len(means) < 2 or len(logvars) < 2:
            return None, None
            
        # Convert to tensors
        means_tensor = torch.cat([m.unsqueeze(0) if m.dim() == 1 else m for m in means])
        logvars_tensor = torch.cat([lv.unsqueeze(0) if lv.dim() == 1 else lv for lv in logvars])
        
        return means_tensor, logvars_tensor