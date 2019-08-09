from networks import ActorCritic_Normal, ActorCritic_Beta
import numpy as np
import torch
import torch.optim as optim

##################################################

class AgentController:
    """An agent controller that interacts with and learns from its environment. The implementation is based on the 
    Proximal Policy Optimization (PPO) algorithm.    
    """
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ##########
    
    def __init__(self,

                 agents_n,

                 frames_n,                 
                 state_size,
                 action_size,
                 
                 action_distribution="normal",
                 hidden_layer_size=128,                 
                 
                 sgd_epochs = 16,
                 minibatch_size = 4096,
                 sgd_inner_iterations = 1,                 
                 
                 gamma = 0.99,
                 gae_lambda = 0.95,                  
                 
                 epsilon_start = 0.2,
                 epsilon_update_factor = 1,
                 
                 objective_weight_critic_start = 0.5,  
                 objective_weight_critic_update_factor = 1,
                 
                 objective_weight_entropy_start = 0,
                 objective_weight_entropy_update_factor = 1,                            
                 
                 learning_rate=3e-4,
                 learning_eps=1e-5,
                 
                 gradient_clip = 5
                ):
        """Initializes an agent controller instance.

        Params
        ======
            agents_n (int): The number of (independent) agents that are learnt from simultaneously.
            frames_n (int): The number of "frames" (or time-steps) that are taken as input simultaneously.
            state_size (int): The dimension of the state space.
            action_size (int): The dimension of the action space.
            action_distribution ("normal" or "beta"): The distribution that is modelled and from which actions are subsequently sampled.
            hidden_layer_size (int): For all networks, the number of nodes to use in a single hidden layer.
            sgd_epochs (int): Per episode, the number of training epochs that are performed. One epoch consists of an iteration through all minibatches.
            minibath_size (int): The number of experiences, out-of-sequence, from all agents, that are used at a time for stochastic gradient descient (SGD).
            sgd_inner_iterations (int): For a given minibatch, the number of times SGD is applied repeatedly at a time.
            gamma (float): The discounting factor to apply per time-step for rewards.
            gae_lambda (float): The Generalized Advantage Estimator's (GAE) "lambda" parameter.
            epsilon_start (float): The starting value of the PPO clipping parameter, "epsilon".
            epsilon_update_factor (float): The factor by which "epsilon" changes every episode.
            objective_weight_critic_start (float): The starting value of the weight in the final objective given to the Critic.
            objective_weight_critic_update_factor (float): The factor by which the weight in the final objective of the Critic changes every episode.
            objective_weight_entropy_start (float): The starting value of the weight in the final objective given to the entropy bonus.
            objective_weight_entropy_update_factor (float): The factor by which the weight in the final objective of the entropy bounus changes per episode.
            learning_rate (float): The learning rate used by the Adam optimizer (jointly for the Actor and Critic networks).
            learning_eps (float): A parameter corresponding to the Adam optimizer's "eps" parameter.
        """
        self.agents_n = agents_n

        self.frames_n = frames_n
        self.state_size = state_size
        self.action_size = action_size

        self.action_distribution=action_distribution
        self.hidden_layer_size=hidden_layer_size

        self.sgd_epochs = sgd_epochs
        self.minibatch_size = minibatch_size
        self.sgd_inner_iterations = sgd_inner_iterations
        
        self.gamma = gamma 
        self.gae_lambda = gae_lambda

        self.epsilon_start = epsilon_start
        self.epsilon_update_factor = epsilon_update_factor
        self.epsilon = self.epsilon_start

        self.objective_weight_critic_start = objective_weight_critic_start
        self.objective_weight_critic_update_factor = objective_weight_critic_update_factor
        self.objective_weight_critic = self.objective_weight_critic_start

        self.objective_weight_entropy_start = objective_weight_entropy_start
        self.objective_weight_entropy_update_factor = objective_weight_entropy_update_factor
        self.objective_weight_entropy = self.objective_weight_entropy_start
        
        self.learning_rate = learning_rate
        self.learning_eps = learning_eps
        
        self.gradient_clip = gradient_clip
                        
        if self.action_distribution == 'normal':
            self.actor_critic = ActorCritic_Normal(self.frames_n, self.state_size, self.action_size, self.hidden_layer_size).to(AgentController.DEVICE)
        elif self.action_distribution == 'beta':
            self.actor_critic = ActorCritic_Beta(self.frames_n, self.state_size, self.action_size, self.hidden_layer_size).to(AgentController.DEVICE)   
        else:
            assert False
                    
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate, eps=self.learning_eps)
        
        self.last_objective = None
                
    ##########
    
    def apply(self, states):
        """Returns detached Actor-Critic predictions (actions, densities, entropies, values) for a given set of states.
        
        Params
        ======
        states: The states from which to make predictions. Here, "states" should not contain a time dimension. Therefore it must either be in "long" form, or arrive one time-step at a time.        
        """
        self.actor_critic.eval()
        with torch.no_grad():
            predictions = self.actor_critic(states)
            to_return = { k: v.detach() for k, v in predictions.items() } # keep it on GPU and as tensors.
        self.actor_critic.train()
        return to_return

    ##########

    def train(self, trajectories):
        """Performs a training step based on a set of trajectories.
        
        Params
        ======
        trajectories: Trajectories of states, actions, densities, values, rewards, and dones produced by the "trajectory_collector" class.
        """
        states = trajectories['states']
        actions = trajectories['actions']
        densities = trajectories['densities']
        values = trajectories['values']
        rewards = trajectories['rewards']
        dones = trajectories['dones']

        returns, advantages = self._estimate_returns_and_advantages(values, rewards, dones)

        # Convert to long form.        
        states_long = states.contiguous().view(states.shape[0] * states.shape[1], states.shape[2], states.shape[3])
        actions_long = actions.view(actions.shape[0] * actions.shape[1], actions.shape[2])
        densities_long = densities.view(densities.shape[0] * densities.shape[1], densities.shape[2])
        densities_long_collapsed = self._collapse_densities(densities_long) 
        
        returns_long = returns.view(returns.shape[0] * returns.shape[1])
        advantages_long = advantages.view(advantages.shape[0] * advantages.shape[1])
                
        ##########

        full_batch_size = states_long.shape[0]        
        minibatches_n = full_batch_size // self.minibatch_size

        for m1 in range(self.sgd_epochs):
            minibatch_base_indexes = torch.Tensor(np.random.permutation(full_batch_size)).long().to(Agent.DEVICE)
            
            for m2 in range(minibatches_n):
                minibatch_indexes = minibatch_base_indexes[(m2 * self.minibatch_size):((m2 + 1) * self.minibatch_size)]
                        
                minibatch_states_long = states_long[minibatch_indexes,:,:]
                minibatch_actions_long = actions_long[minibatch_indexes,:]
                minibatch_densities_long_collapsed = densities_long_collapsed[minibatch_indexes]

                minibatch_returns_long = returns_long[minibatch_indexes]
                minibatch_advantages_long = advantages_long[minibatch_indexes]
                
                for m3 in range(self.sgd_inner_iterations):
                
                    new_predictions = self.actor_critic(minibatch_states_long, minibatch_actions_long)

                    minibatch_new_densities_long = new_predictions['densities']
                    minibatch_new_densities_long_collapsed = self._collapse_densities(minibatch_new_densities_long)
                    minibatch_new_entropies_long = new_predictions['entropies']
                    minibatch_new_entropies_long_collapsed = self._collapse_entropies(minibatch_new_entropies_long)

                    minibatch_new_values_long = new_predictions['values'][:,0]

                    ratios = minibatch_new_densities_long_collapsed / minibatch_densities_long_collapsed

                    ratios_advantages = ratios * minibatch_advantages_long   
                    ratios_clipped = torch.clamp(ratios, min=1 - self.epsilon, max=1 + self.epsilon)

                    ratios_clipped_advantages = ratios_clipped * minibatch_advantages_long

                    clipped_component = torch.min(ratios_advantages, ratios_clipped_advantages)

                    ##########

                    objective_component_1 = -torch.mean(clipped_component)  # Negative to ensure maximisation.                    
                    objective_component_2 = self.objective_weight_critic * torch.mean((minibatch_returns_long - minibatch_new_values_long).pow(2))  # Positive to ensure minimisation.               
                    objective_component_3 = -self.objective_weight_entropy * torch.mean(minibatch_new_entropies_long_collapsed)  # Negative to ensure maximisation.

                    objective = objective_component_1 + objective_component_2 + objective_component_3

                    self.optimizer.zero_grad()
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.gradient_clip)
                    self.optimizer.step()

        self._update_parameters()

    ##########
    
    def save(self, path):
        """Saves the Actor-Critic network's weights to the given path.
        
        path (str): The path to which to save the Actor-Critic's network weights.
        """
        torch.save(self.actor_critic.state_dict(), path)
    
    ##########    
    
    def _estimate_returns_and_advantages(self, values, rewards, dones):
        """Discounts rewards into returns; uses GAE to calculate standardized advantages.
        
        Params
        ======
        values: The Critic values of the trajectory.
        rewards: The rewards of the trajectory.
        dones: The done-values of the trajectory.
        
        Returns
        =======
        returns: The discounted rewards.
        advantages: The standardized GAE advantages.        
        
        Note: Inspired by https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/PPO_agent.py
        """
        steps_n = rewards.shape[1] # Note: Don't use 'values' to get "steps_n": it has one extra entry in dimension 1.

        returns = torch.zeros((self.agents_n, steps_n)).float().to(AgentController.DEVICE)
        advantages = torch.zeros((self.agents_n, steps_n)).float().to(AgentController.DEVICE)

        returns_accumulator = values[:,steps_n] # Note: One step beyond what the structures have.
        advantage_accumulator = torch.zeros((self.agents_n)).float().to(AgentController.DEVICE)

        for t in reversed(range(steps_n)):
            returns_accumulator = rewards[:,t] + self.gamma * (1 - dones[:,t]) * returns_accumulator
            td_error = rewards[:,t] + self.gamma * (1 - dones[:,t]) * values[:,t + 1] - values[:,t]
            advantage_accumulator  = advantage_accumulator * self.gae_lambda * self.gamma * (1 - dones[:,t]) + td_error
            returns[:,t] = returns_accumulator
            advantages[:,t] = advantage_accumulator

        # Standardise the advantages (on this problem, it seems to work without this step as well).
        # Note: This standardisation occurs over all agents simultaneously.
        advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
                
        return returns, advantages
    
    ##########    
    
    def _collapse_densities(self, densities):
        """Collapses densities over the action space.
        
        Params
        ======
        densities: The uncollapsed densities.
        """
        return torch.prod(densities, dim=-1)

    ##########    
    
    def _collapse_entropies(self, entropies):
        """Collapses entropies over the action space.
        
        Params
        ======
        entropies: The uncollapsed entropies.
        """
        return torch.sum(entropies, dim=-1)

    ##########
        
    def _update_parameters(self):
        """Updates some parameters. To be called after every episode.
        """
        self.epsilon *= self.epsilon_update_factor
        self.objective_weight_critic *= self.objective_weight_critic_update_factor 
        self.objective_weight_entropy *= self.objective_weight_entropy_update_factor         
