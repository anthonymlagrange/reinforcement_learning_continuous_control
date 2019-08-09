# from moment_tracker import MomentTracker    
import numpy as np
import torch

class TrajectoryCollector:
    """Instances of this class collect trajectories for one or more independent agents from an environment.
    """
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SMALL_POSITIVE_NUMBER = 1.0e-9
    
    def __init__(self, env, brain_name, agent_controller):
        """Initializes an instance of the class.
        
        Params
        ======
        env: The environment.
        brain_name (str): The brain name to use in the environment.
        agent_controller: The agent_controller instance.        
        """
        
        self.env = env
        self.env_info = None
        self.brain_name = brain_name
        self.agent_controller = agent_controller
        
        #self.state_moment_tracker = MomentTracker(name="state_tracker", dim=self.agent_controller.state_size, verbose_at=1000000)
        #self.action_moment_tracker = MomentTracker(name="action_tracker", dim=self.agent_controller.action_size, verbose_at=1000000)
        #self.reward_moment_tracker = MomentTracker(name="rewards_tracker", dim=self.agent_controller.agents_n, verbose_at=1000000)

    def reset_environment(self):
        """Resets the environment.
        """
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]  
        # Note: Do *not* reset the moment trackers.
                    
    def generate(self, steps_n):
        """Generates and returns a trajectory.
        
        Params
        ======
        steps_n (int): The trajectory is for up to "steps_n" steps. If any agent is reported to be done earlier, the trajectory stops at that point.
        """
        trajectories = {}
        trajectories['states'] = torch.zeros((self.agent_controller.agents_n, steps_n + 1, self.agent_controller.frames_n, self.agent_controller.state_size)).to(TrajectoryCollector.DEVICE)  # Note the initial additional time-step.
        trajectories['actions'] = torch.zeros((self.agent_controller.agents_n, steps_n, self.agent_controller.action_size)).to(TrajectoryCollector.DEVICE)
        trajectories['densities'] = torch.zeros((self.agent_controller.agents_n, steps_n, self.agent_controller.action_size)).to(TrajectoryCollector.DEVICE)
        trajectories['entropies'] = torch.zeros((self.agent_controller.agents_n, steps_n, self.agent_controller.action_size)).to(TrajectoryCollector.DEVICE)
        trajectories['values'] = torch.zeros((self.agent_controller.agents_n, steps_n + 1)).to(TrajectoryCollector.DEVICE)  # Note the additional time-step.
        trajectories['rewards'] = torch.zeros((self.agent_controller.agents_n, steps_n)).to(TrajectoryCollector.DEVICE)
        trajectories['dones'] = torch.zeros((self.agent_controller.agents_n, steps_n)).to(TrajectoryCollector.DEVICE)

        for t in range(steps_n):
            self._add_states(trajectories['states'], t)
            
            predictions = self.agent_controller.apply(trajectories['states'][:,t,:,:])
            
            trajectories['actions'][:,t,:] = predictions['actions']
            trajectories['densities'][:,t,:] = predictions['densities']
            trajectories['entropies'][:,t,:] = predictions['entropies']
            trajectories['values'][:,t] = predictions['values'][:,0]

            self.env_info = self.env.step(predictions['actions'].cpu().numpy())[self.brain_name] 
            
            rewards = torch.Tensor(self.env_info.rewards).float().to(TrajectoryCollector.DEVICE)
            trajectories['rewards'][:,t] = rewards

            dones = self.env_info.local_done
            trajectories['dones'][:,t] = torch.Tensor(dones).float().to(TrajectoryCollector.DEVICE)

            if np.any(dones):
                break

        t += 1

        # Calculate one additional set of values:
        self._add_states(trajectories['states'], t)        
        trajectories['values'][:,t] = self.agent_controller.apply(trajectories['states'][:,t,:,:])['values'][:,0]
        
        # Truncate:
        trajectories['states'] =  trajectories['states'][:,:t,:,:] 
        trajectories['actions'] = trajectories['actions'][:,:t,:]
        trajectories['densities'] = trajectories['densities'][:,:t,:]
        trajectories['entropies'] = trajectories['entropies'][:,:t,:]        
        trajectories['values'] = trajectories['values'][:,:(t+1)]  # Note: "values" is one larger.
        trajectories['rewards'] = trajectories['rewards'][:,:t]
        trajectories['dones'] = trajectories['dones'][:,:t]

        return trajectories

    def _add_states(self, current_states, t):
        """A convenience function that adds states into the trajectory, taking care of capturing "self.agent_controller.frames_n" frames back.
        """
        raw_states = self.env_info.vector_observations
        current_states[:,t,self.agent_controller.frames_n - 1,:] = torch.from_numpy(raw_states).float().to(TrajectoryCollector.DEVICE)

        if t > 0:  # Take care of multiple frames, if any.
            for f in range(self.agent_controller.frames_n - 1):
                current_states[:,t,f,:] = current_states[:,t-1,f+1,:]
