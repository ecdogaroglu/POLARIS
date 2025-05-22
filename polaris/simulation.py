"""
Core simulation logic for POLARIS experiments.
"""

import numpy as np
import time
import torch
from tqdm import tqdm
from polaris.utils.utils import (
    create_output_directory,
    calculate_observation_dimension,
    encode_observation, 
    setup_random_seeds,
    calculate_theoretical_bounds,
    write_config_file,
    flatten_episodic_metrics,
    save_final_models,
    set_metrics,
    reset_agent_internal_states,
    update_total_rewards,
    select_agent_actions,
    update_progress_display,
    store_transition_in_buffer,
    load_agent_models)
from polaris.utils.metrics import (
    initialize_metrics, 
    update_metrics, 
    calculate_agent_learning_rates_from_metrics,
    prepare_serializable_metrics,
    save_metrics_to_file
)
from polaris.visualizations.plotting import generate_plots
from polaris.agent.polaris_agent import POLARISAgent
from polaris.agent.networks import TemporalGNN
from polaris.agent.replay_buffer import ReplayBuffer

def run_agents(env, args, training=True, model_path=None):
    """
    Run POLARIS agents in the social learning environment.
    
    Args:
        env: The social learning environment
        args: Command-line arguments
        training: Whether to train the agents (True) or just evaluate (False)
        model_path: Path to load models from (optional)
    
    Returns:
        learning_rates: Dictionary of learning rates for each agent
        serializable_metrics: Dictionary of metrics for JSON serialization
    """
    # Setup directory
    output_dir = create_output_directory(args, env, training)
    
    # Initialize agents and metrics
    obs_dim = calculate_observation_dimension(env)
    agents = initialize_agents(env, args, obs_dim)
    load_agent_models(agents, model_path, env.num_agents, training=training)
    metrics = initialize_metrics(env, args, training)
    
    # Store agents for potential SI visualization
    if hasattr(args, 'visualize_si') and args.visualize_si and training:
        from polaris.visualizations.si_visualizer import create_si_visualizations
        create_si_visualizations(agents, output_dir)

    # Calculate and display theoretical bounds
    theoretical_bounds = calculate_theoretical_bounds(env)
    #display_theoretical_bounds(theoretical_bounds)
    
    replay_buffers = initialize_replay_buffers(agents, args, obs_dim)

    # Write configuration if training
    if training:
        write_config_file(args, env, theoretical_bounds, output_dir)

    print(f"Running {args.num_episodes} episode(s) with {args.horizon} steps per episode")
    
    # Initialize episodic metrics to store each episode separately
    episodic_metrics = {
        'episodes': []
    }
    
    # Episode loop
    for episode in range(args.num_episodes):
        # Set a different seed for each episode based on the base seed
        episode_seed = args.seed + episode
        setup_random_seeds(episode_seed, env)
        print(f"\nStarting episode {episode+1}/{args.num_episodes} with seed {episode_seed}")
        
        # Initialize fresh metrics for this episode
        metrics = initialize_metrics(env, args, training)
        
        # Run simulation for this episode
        observations, episode_metrics = run_simulation(
            env, agents, replay_buffers, metrics, args,
            output_dir, training
        )
        
        # Store this episode's metrics separately
        episodic_metrics['episodes'].append(episode_metrics)
    
    # Create a flattened version of metrics
    combined_metrics = flatten_episodic_metrics(episodic_metrics, env.num_agents)
    
    # Process results
    learning_rates = calculate_agent_learning_rates_from_metrics(combined_metrics)
    
    # Create SI visualizations after training is complete
    if hasattr(args, 'visualize_si') and args.visualize_si and training:
        from polaris.visualizations.si_visualizer import create_si_visualizations
        print("\n===== FINAL SI STATE (AFTER TRAINING) =====")
        create_si_visualizations(agents, output_dir)
    
    # Save metrics and models
    serializable_metrics = prepare_serializable_metrics(
        combined_metrics, learning_rates, theoretical_bounds, args.horizon, training
    )
    
    # Also save the episodic metrics for more detailed analysis
    episodic_serializable_metrics = {
        'episodic_data': episodic_metrics,
        'learning_rates': learning_rates,
        'theoretical_bounds': theoretical_bounds,
        'episode_length': args.horizon,
        'num_episodes': args.num_episodes
    }
    
    save_metrics_to_file(serializable_metrics, output_dir, training)
    save_metrics_to_file(episodic_serializable_metrics, output_dir, training, filename='episodic_metrics.json')
    
    if training and args.save_model:
        save_final_models(agents, output_dir)
    
    # Generate plots with LaTeX style if requested
    generate_plots(
        combined_metrics,
        env, 
        args, 
        output_dir, 
        training, 
        episodic_metrics,
        use_latex=args.use_tex if hasattr(args, 'use_tex') else False
    )
    
    return episodic_metrics, serializable_metrics

def run_simulation(env, agents, replay_buffers, metrics, args, output_dir, training):
    """Run the main simulation loop."""
    mode_str = "training" if training else "evaluation"
    
    print(f"Starting {mode_str} for {args.horizon} steps...")
    start_time = time.time()
    
    # Initialize environment and agents
    observations = env.initialize()
    total_rewards = np.zeros(env.num_agents) if training else None
    
    if hasattr(env, 'safe_payoff'):
        # Print the true state
        if env.true_state == 0:
            print(f"True state is bad. Drift rate: {env.drift_rates[env.true_state]} Jump rate: {env.jump_rates[env.true_state]} Jump size: {env.jump_sizes[env.true_state]}")
        else:
            print(f"True state is good. Drift rate: {env.drift_rates[env.true_state]} Jump rate: {env.jump_rates[env.true_state]} Jump size: {env.jump_sizes[env.true_state]}")    
    else:
        print(f"True state is {env.true_state}")

    # If training and using SI, set the current true state for all agents
    if training and hasattr(args, 'use_si') and args.use_si:
        current_true_state = env.true_state
        for agent_id, agent in agents.items():
            if hasattr(agent, 'use_si') and agent.use_si:
                agent.current_true_state = current_true_state
                # Set the current task in the SI trackers
                if hasattr(agent, 'belief_si') and hasattr(agent, 'policy_si'):
                    agent.belief_si.set_task(current_true_state)
                    agent.policy_si.set_task(current_true_state)
                    # Mark that this agent has path integrals calculated so the SI loss will be applied
                    agent.path_integrals_calculated = True
                print(f"Set current true state {current_true_state} for agent {agent_id}")
        
    # Set global metrics for access in other functions
    set_metrics(metrics)
    
    # Reset and initialize agent internal states
    reset_agent_internal_states(agents)
    
    # Set agents to appropriate mode
    for agent_id, agent in agents.items():
        if training:
            agent.set_train_mode()
        else:
            agent.set_eval_mode()
    
    # Extract environment parameters for MPE calculation if this is Strategic Experimentation
    env_params = {}
    if hasattr(env, 'safe_payoff'):
        env_params = {
            'safe_payoff': env.safe_payoff,
            'drift_rates': env.drift_rates,
            'jump_rates': env.jump_rates,
            'jump_sizes': env.jump_sizes,
            'background_informativeness': env.background_informativeness,
            'num_agents': env.num_agents,
            'true_state': env.true_state
        }
    
    # Main simulation loop
    steps_iterator = tqdm(range(args.horizon), desc="Training" if training else "Evaluating")
    for step in steps_iterator:
        # Get agent actions
        actions, action_probs = select_agent_actions(agents, metrics)
        
        # Collect policy means and standard deviations for continuous actions
        policy_means = []
        policy_stds = []
        
        # Collect agent beliefs for MPE calculation
        agent_beliefs = {}
        
        if hasattr(args, 'continuous_actions') and args.continuous_actions:
            for agent_id, agent in agents.items():
                if hasattr(agent, 'action_mean') and hasattr(agent.policy, 'forward'):
                    # Get policy parameters directly
                    with torch.no_grad():
                        mean, log_std = agent.policy(agent.current_belief, agent.current_latent)
                        std = torch.exp(log_std)
                    
                    policy_means.append(mean.item())
                    policy_stds.append(std.item())
                    
                    # Extract agent's belief about the state
                    if hasattr(agent, 'current_belief_distribution') and agent.current_belief_distribution is not None:
                        # For binary state (common case), the belief about good state is the probability assigned to state 1
                        if agent.current_belief_distribution.shape[-1] == 2:
                            agent_beliefs[agent_id] = agent.current_belief_distribution[0, 1].item()
                        else:
                            # For multi-state cases, use a weighted average
                            belief_weights = torch.arange(agent.current_belief_distribution.shape[-1], 
                                                          device=agent.current_belief_distribution.device).float()
                            belief_weights = belief_weights / (agent.current_belief_distribution.shape[-1] - 1)  # Normalize to [0,1]
                            agent_beliefs[agent_id] = torch.sum(agent.current_belief_distribution * belief_weights, dim=-1).item()
                else:
                    # Fallback if action_mean not stored
                    policy_means.append(0.5)  # Default middle value
                    policy_stds.append(0.1)  # Default small std
                    agent_beliefs[agent_id] = 0.5  # Default middle belief
        
        # Take environment step
        next_observations, rewards, done, info = env.step(actions, action_probs)
        
        # Add policy distribution parameters to info
        if hasattr(args, 'continuous_actions') and args.continuous_actions:
            info['policy_means'] = policy_means
            info['policy_stds'] = policy_stds
            info['agent_beliefs'] = agent_beliefs
            info['env_params'] = env_params
        
        # Update rewards if training
        if training and rewards:
            update_total_rewards(total_rewards, rewards)
        
        # Update agent states and store transitions
        update_agent_states(
            agents, observations, next_observations, actions, rewards, 
            replay_buffers, metrics, env, args, training, step
        )
        
        # Update observations for next step
        observations = next_observations
        
        # For continuous actions in Strategic Experimentation env, add allocations to info
        if hasattr(args, 'continuous_actions') and args.continuous_actions and hasattr(env, 'safe_payoff'):
            # Add allocations to info for metrics tracking
            if 'allocations' not in info:
                info['allocations'] = actions
        
        # Store and process metrics
        update_metrics(
            metrics, 
            info, 
            actions, 
            action_probs
        )
        
        # Update progress display
        update_progress_display(steps_iterator, info, total_rewards, step, training)
        
        # Save models periodically if training
        #if training and args.save_model and (step + 1) % max(1, args.horizon // 5) == 0:
        #    save_checkpoint_models(agents, output_dir, step)
        
        if done:
            # Check if we have a new true state for SI
            if training and hasattr(args, 'use_si') and args.use_si:
                current_true_state = env.true_state
                print(f"Current true state: {current_true_state}")
                # Check if this is a new true state for any agent
                for agent_id, agent in agents.items():
                    if hasattr(agent, 'use_si') and agent.use_si and hasattr(agent, 'seen_true_states'):
                        if current_true_state not in agent.seen_true_states:
                            # We have a new true state, register the previous task and set the new one
                            if hasattr(agent, 'belief_si') and hasattr(agent, 'policy_si'):
                                # Register completed task for both networks
                                agent.belief_si.register_task()
                                agent.policy_si.register_task()
                                
                                # Set new task
                                agent.belief_si.set_task(current_true_state)
                                agent.policy_si.set_task(current_true_state)
                                
                                # Store task-specific trackers for visualization
                                if hasattr(agent, 'state_belief_si_trackers'):
                                    # Create clones of the trackers for visualization
                                    agent.state_belief_si_trackers[current_true_state] = agent._clone_si_tracker(agent.belief_si)
                                    agent.state_policy_si_trackers[current_true_state] = agent._clone_si_tracker(agent.policy_si)
                                
                                print(f"Registered completed task and set new true state {current_true_state} for agent {agent_id}")
                            
                        # Add current true state to the set of seen states
                        agent.seen_true_states.add(current_true_state)
                        # Update the current true state
                        agent.current_true_state = current_true_state

            break
    
    # Display completion time
    total_time = time.time() - start_time
    print(f"{mode_str.capitalize()} completed in {total_time:.2f} seconds")
    
    return observations, metrics

def update_agent_states(agents, observations, next_observations, actions, rewards, 
                        replay_buffers, metrics, env, args, training, step):
    """Update agent states and store transitions in replay buffer."""
    
    # Check if we're using continuous actions
    continuous_actions = hasattr(args, 'continuous_actions') and args.continuous_actions
    
    for agent_id, agent in agents.items():
        # Get current and next observations
        obs_data = observations[agent_id]
        next_obs_data = next_observations[agent_id]
        
        # Extract signals and neighbor actions based on environment type
        if 'signal' in obs_data:
            # Social Learning Environment format
            signal = obs_data['signal']
            next_signal = next_obs_data['signal']
            neighbor_actions = obs_data['neighbor_actions']
            next_neighbor_actions = next_obs_data['neighbor_actions']
        elif 'background_signal' in obs_data:
            # Strategic Experimentation Environment format
            # Use background signal as the observation signal
            signal = int(obs_data['background_signal'] > 0)  # Convert to binary signal: 1 if positive, 0 if negative
            next_signal = int(next_obs_data['background_signal'] > 0)
            # Get allocations instead of discrete actions
            neighbor_allocations = obs_data.get('neighbor_allocations', {})
            next_neighbor_allocations = next_obs_data.get('neighbor_allocations', {})
            # Handle None values by using empty dictionaries instead
            if continuous_actions:
                # Use raw allocation values
                neighbor_actions = {} if neighbor_allocations is None else neighbor_allocations
                next_neighbor_actions = {} if next_neighbor_allocations is None else next_neighbor_allocations
            else:
                # Convert to binary actions
                neighbor_actions = {} if neighbor_allocations is None else {k: int(v > 0.5) for k, v in neighbor_allocations.items()}
                next_neighbor_actions = {} if next_neighbor_allocations is None else {k: int(v > 0.5) for k, v in next_neighbor_allocations.items()}

        # Encode observations
        signal_encoded, actions_encoded = encode_observation(
            signal=signal,
            neighbor_actions=neighbor_actions,
            num_agents=env.num_agents,
            num_states=env.num_states,
            continuous_actions=continuous_actions
        )
        next_signal_encoded, _ = encode_observation(
            signal=next_signal,
            neighbor_actions=next_neighbor_actions,
            num_agents=env.num_agents,
            num_states=env.num_states,
            continuous_actions=continuous_actions
        )

        # Get current belief and latent states (before observation update)
        belief = agent.current_belief.detach().clone()  # Make a copy to ensure we have the pre-update state
        latent = agent.current_latent.detach().clone()

        # Update agent belief state
        next_belief, next_dstr = agent.observe(signal_encoded, actions_encoded)
        # Infer latent state for next observation
        # This ensures we're using the correct latent state for the next observation
        next_latent = agent.infer_latent(
            signal_encoded,
            actions_encoded,
            rewards[agent_id] if isinstance(rewards[agent_id], float) else rewards[agent_id]['total'],
            next_signal_encoded
        )
            
        # Store internal states for visualization if requested (for both training and evaluation)

        if args.plot_internal_states and 'belief_states' in metrics:
            current_belief = agent.get_belief_state()
            current_latent = agent.get_latent_state()
            metrics['belief_states'][agent_id].append(current_belief.detach().cpu().numpy())
            metrics['latent_states'][agent_id].append(current_latent.detach().cpu().numpy())
            
                
            # Store opponent belief distribution if available
            opponent_belief_distribution = agent.get_opponent_belief_distribution()
            metrics['opponent_belief_distributions'][agent_id].append(opponent_belief_distribution.detach().cpu().numpy())
        
        # Store belief distribution if available
        belief_distribution = agent.get_belief_distribution()
        metrics['belief_distributions'][agent_id].append(belief_distribution.detach().cpu().numpy())
        
        # Store transition in replay buffer if training
        if training and agent_id in replay_buffers:
            
            # Get mean and logvar from inference
            mean, logvar = agent.get_latent_distribution_params()
            
            # Get reward value (handle both scalar and dictionary cases)
            reward_value = rewards[agent_id]['total'] if isinstance(rewards[agent_id], dict) else rewards[agent_id]
            
            # Store transition
            store_transition_in_buffer(
                replay_buffers[agent_id],
                signal_encoded,
                actions_encoded,
                belief,
                latent,
                actions[agent_id],
                reward_value,
                next_signal_encoded,
                next_belief,
                next_latent,
                mean,
                logvar
            )
            
            # Update networks if enough samples
            if len(replay_buffers[agent_id]) > args.batch_size and step % args.update_interval == 0:
                # Sample a batch from the replay buffer
                batch = replay_buffers[agent_id].sample(args.batch_size)
                # Update network parameters
                agent.update(batch)

def initialize_agents(env, args, obs_dim):
    """Initialize POLARIS agents."""
    print(f"Initializing {env.num_agents} agents{' for evaluation' if args.eval_only else ''}...")
    
    # Log if using GNN
    if args.use_gnn:
        print(f"Using Graph Neural Network with {args.gnn_layers} layers, {args.attn_heads} attention heads, and temporal window of {args.temporal_window}")
    else:
        print("Using traditional encoder-decoder inference module")
    
    # Log if excluding final layers from SI
    if hasattr(args, 'si_exclude_final_layers') and args.si_exclude_final_layers and hasattr(args, 'use_si') and args.use_si:
        print("Excluding final layers from Synaptic Intelligence protection")
    
    # Log if using continuous actions
    if hasattr(args, 'continuous_actions') and args.continuous_actions:
        print("Using continuous action space for strategic experimentation")
        
    agents = {}
    
    for agent_id in range(env.num_agents):
        # Determine action dimension based on environment and action space type
        if hasattr(args, 'continuous_actions') and args.continuous_actions:
            # For continuous actions, we use 1 dimension (allocation between 0 and 1)
            action_dim = 1  
        else:
            # For discrete actions, we use num_states dimensions
            action_dim = env.num_states
            
        agent = POLARISAgent(
            agent_id=agent_id,
            num_agents=env.num_agents,
            num_states=env.num_states,
            observation_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            belief_dim=args.belief_dim,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            entropy_weight=args.entropy_weight,
            kl_weight=args.kl_weight,
            device=args.device,
            buffer_capacity=args.buffer_capacity,
            max_trajectory_length=args.horizon,
            use_gnn=args.use_gnn,
            use_si=args.use_si if hasattr(args, 'use_si') else False,
            si_importance=args.si_importance if hasattr(args, 'si_importance') else 100.0,
            si_damping=args.si_damping if hasattr(args, 'si_damping') else 0.1,
            si_exclude_final_layers=args.si_exclude_final_layers if hasattr(args, 'si_exclude_final_layers') else False,
            continuous_actions=args.continuous_actions if hasattr(args, 'continuous_actions') else False
        )
        
        # If using GNN, update the inference module with the specified parameters
        if args.use_gnn and hasattr(agent, 'inference_module'):
            agent.inference_module = TemporalGNN(
                hidden_dim=args.hidden_dim,
                action_dim=action_dim,
                latent_dim=args.latent_dim,
                num_agents=env.num_agents,
                device=args.device,
                num_belief_states=env.num_states,
                num_gnn_layers=args.gnn_layers,
                num_attn_heads=args.attn_heads,
                dropout=0.1,
                temporal_window_size=args.temporal_window
            ).to(args.device)
            
            # Update the optimizer to use the new inference module
            agent.inference_optimizer = torch.optim.Adam(
                agent.inference_module.parameters(),
                lr=args.learning_rate
            )
            
        agents[agent_id] = agent
            
    return agents

def initialize_replay_buffers(agents, args, obs_dim):
    """Initialize replay buffers for training."""
    replay_buffers = {}
    
    for agent_id in agents:
        replay_buffers[agent_id] = ReplayBuffer(
            capacity=args.buffer_capacity,
            observation_dim=obs_dim,
            belief_dim=args.belief_dim,
            latent_dim=args.latent_dim,
            device=args.device,
            sequence_length=8  # Default sequence length for sampling
        )
    return replay_buffers