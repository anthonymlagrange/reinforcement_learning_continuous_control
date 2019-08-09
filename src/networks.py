import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic_Normal(nn.Module):
    """An Actor-Critic network that uses the Normal distribution to sample actions.
    """        

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SMALL_POSITIVE_NUMBER = 1.0e-9

    def __init__(self, frames_n, state_size, action_size, hidden_layer_size):
        """Initializes an Actor-Critic network based on the Normal distribution.

        Params
        ======
        frames_n (int): The number of "frames" of state that are taken into account simultaneously.
        state_size (int): The dimension of the state space.
        action_size (int): The dimension of the action space.
        hidden_layer_size (int): The number of nodes to use in the single hidden layer.
        """
        super(ActorCritic_Normal, self).__init__()
        
        self.frames_n = frames_n        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_size = hidden_layer_size

        self.actor_layer_1 = nn.Linear(in_features=self.frames_n * self.state_size, out_features=self.hidden_layer_size)
        self.actor_layer_2 = nn.Linear(in_features=self.hidden_layer_size, out_features=self.hidden_layer_size)
        self.actor_layer_3 = nn.Linear(in_features=self.hidden_layer_size, out_features=self.action_size)
        
        # Here, the sigma values do not depend directly on the states in a forward step. Only indirectly in a backward step.
        # The starting values are taken to be 1.
        self.actor_sigmas_parameters = nn.Parameter(torch.ones(1, self.action_size))
        
        self.critic_layer_1 = nn.Linear(in_features=self.frames_n * self.state_size, out_features=self.hidden_layer_size)
        self.critic_layer_2 = nn.Linear(in_features=self.hidden_layer_size, out_features=self.hidden_layer_size)
        self.critic_layer_3 = nn.Linear(in_features=self.hidden_layer_size, out_features=1)

    def forward(self, states, actions_previous=None):
        """Performs a forward step.

        Params
        ======
        states: The states from which to make predictions. Here, "states" should not contain a time dimension. Therefore it must either be in "long" form, or arrive one time-step at a time.
        actions_previous: Optionally, if a replay, provide the previous actions. This results in an update of the densities, entropies, and values based on the previous actions.

        Returns
        =======
        actions: The resultant actions.
        densities: The resultant densities.
        entropies: The resultant entropies.
        values: The resultant values.
        """
        x = states.view(states.shape[0], self.frames_n * self.state_size)
        
        # ACTOR
        x_actor_mus = F.relu(self.actor_layer_1(x))
        x_actor_mus = F.relu(self.actor_layer_2(x_actor_mus))
        x_actor_mus = torch.tanh(self.actor_layer_3(x_actor_mus))

        distribution = torch.distributions.normal.Normal(loc=x_actor_mus, scale=self.actor_sigmas_parameters)
        actions = actions_previous if actions_previous is not None else distribution.sample() 
        # actions = torch.clamp(actions, -1, 1)  # Note: This is one approach, if necessary. Another is to use a Beta distribution
            # instead of a Normal distribution (see below).
        densities = torch.exp(distribution.log_prob(actions))
        entropies = distribution.entropy()
        
        # CRITIC
        x_critic = F.relu(self.critic_layer_1(x))
        x_critic = F.relu(self.critic_layer_2(x_critic))
        values = self.critic_layer_3(x_critic)
        
        return {
            'actions': actions,
            'densities': densities,
            'entropies': entropies,            
            'values': values
        }
        
class ActorCritic_Beta(nn.Module):
    """An Actor-Critic network that uses the Beta distribution to sample actions.
    """                    
        
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SMALL_POSITIVE_NUMBER = 1.0e-9
    
    def __init__(self, frames_n, state_size, action_size, hidden_layer_size):
        """Initializes an Actor-Critic network based on the Beta distribution.

        Params
        ======
        frames_n (int): The number of "frames" of state that are taken into account simultaneously.
        state_size (int): The dimension of the state space.
        action_size (int): The dimension of the action space.
        hidden_layer_size (int): The number of nodes to use in the single hidden layer.
        """
        super(ActorCritic_Beta, self).__init__()
              
        self.frames_n = frames_n        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_size = hidden_layer_size
        
        # ACTOR
        self.actor_alphas_layer_1 = nn.Linear(in_features=self.frames_n * self.state_size, out_features=self.hidden_layer_size)
        self.actor_alphas_layer_2 = nn.Linear(in_features=self.hidden_layer_size, out_features=self.hidden_layer_size)
        self.actor_alphas_layer_3 = nn.Linear(in_features=self.hidden_layer_size, out_features=self.action_size)
        
        self.actor_betas_layer_1 = nn.Linear(in_features=self.frames_n * self.state_size, out_features=self.hidden_layer_size)
        self.actor_betas_layer_2 = nn.Linear(in_features=self.hidden_layer_size, out_features=self.hidden_layer_size)
        self.actor_betas_layer_3 = nn.Linear(in_features=self.hidden_layer_size, out_features=self.action_size)
        
        # CRITIC        
        self.critic_layer_1 = nn.Linear(in_features=self.frames_n * self.state_size, out_features=self.hidden_layer_size)
        self.critic_layer_2 = nn.Linear(in_features=self.hidden_layer_size, out_features=self.hidden_layer_size)
        self.critic_layer_3 = nn.Linear(in_features=self.hidden_layer_size, out_features=1)

    def forward(self, states, actions_previous=None):
        """Performs a forward step.

        Params
        ======
        states: The states from which to make predictions. Here, "states" should not contain a time dimension. Therefore it must either be in "long" form, or arrive one time-step at a time.
        actions_previous: Optionally, if a replay, provide the previous actions. This results in an update of the densities, entropies, and values based on the previous actions.

        Returns
        =======
        actions: The resultant actions.
        densities: The resultant densities.
        entropies: The resultant entropies.
        values: The resultant values.
        """    
        x = states.view(states.shape[0], self.frames_n * self.state_size)
        
        # ACTOR
        x_actor_alphas = F.relu(self.actor_alphas_layer_1(x))
        x_actor_alphas = F.relu(self.actor_alphas_layer_2(x_actor_alphas))
        x_actor_alphas = F.softplus(self.actor_alphas_layer_3(x_actor_alphas)) + 1.  # To get to the interval [1; Inf).

        x_actor_betas = F.relu(self.actor_betas_layer_1(x))
        x_actor_betas = F.relu(self.actor_betas_layer_2(x_actor_betas))
        x_actor_betas = F.softplus(self.actor_betas_layer_3(x_actor_betas)) + 1.  # To get to the interval [1; Inf).
        
        distribution = torch.distributions.beta.Beta(concentration1=x_actor_alphas, concentration0=x_actor_betas)
        raw_actions = actions_previous * 0.5 + 0.5 if actions_previous is not None else distribution.sample()  # To return to the Beta interval, [0, 1], for now.
        densities = torch.exp(distribution.log_prob(raw_actions))
        actions = (raw_actions - 0.5) * 2  # Finally back to the action interval, [-1, -1].
        entropies = distribution.entropy()
          
        # CRITIC
        x_critic = F.relu(self.critic_layer_1(x))
        x_critic = F.relu(self.critic_layer_2(x_critic))
        values = self.critic_layer_3(x_critic)
            
        return {
            'actions': actions,
            'densities': densities,
            'entropies': entropies,            
            'values': values
        }
