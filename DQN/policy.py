"""RL Policy classes.
"""
import numpy as np
import attr
import random

class Policy:
    """Base class representing an MDP policy.

    Policies are used by the agent to choose actions.

    Policies are designed to be stacked to get interesting behaviors
    of choices. For instances in a discrete action space the lowest
    level policy may take in Q-Values and select the action index
    corresponding to the largest value. If this policy is wrapped in
    an epsilon greedy policy then with some probability epsilon, a
    random action will be chosen.
    """

    def select_action(self, **kwargs):
        """Used by agents to select actions.

        - Output
          - Any:
            An object representing the chosen action. Type depends on
            the hierarchy of policy instances.
        """
        raise NotImplementedError('This method should be overriden.')


class UniformRandomPolicy(Policy):
    """Uniformly randomly Chooses a discrete action. 


    - Parameters
      - num_actions: int
        Number of actions to choose from. Must be > 0.

    - Raises
      - ValueError:
        If num_actions <= 0
    """

    def __init__(self, num_actions):
        assert num_actions >= 1
        self.num_actions = num_actions

    def select_action(self, **kwargs):
        """Return a random action index.

        - Output
          - int
            Action index in range [0, num_actions)
        """
        return np.random.randint(0, self.num_actions)

    def get_config(self):  
        return {'num_actions': self.num_actions}


class GreedyPolicy(Policy):
    """Always returns the best action according to Q-values.

    This is a pure exploitation policy.
    """

    def select_action(self, q_values, **kwargs):  
        """Return a random action index.

        - Input
          - q_values: np.array
            Q-values for all actions

        - Output
          - int
            Action index in range [0, num_actions)
        """
 
        return np.argmax(q_values)


class GreedyEpsilonPolicy(Policy):
    """Selects greedy action or with some probability a random action.

    Standard greedy-epsilon implementation. With probability epsilon
    choose a random action. Otherwise choose the greedy action.

    - Parameters
      - num_actions: int
        Number of actions to choose from. Must be > 0.
      - epsilon: float
       - Initial probability of choosing a random action. 
    """
    def __init__(self, epsilon, num_actions):
        self.epsilon = epsilon
        assert num_actions >= 1
        self.num_actions = num_actions

    def select_action(self, q_values, **kwargs):
        """Run Greedy-Epsilon for the given Q-values.

        - Input
          - q_values: np.array
            Q-values for all actions

        - Output
          - int
            The action index chosen.
        """
        x = random.uniform(0, 1) 
        if x > self.epsilon:
            return np.argmax(q_values)
        return np.random.randint(0, self.num_actions)
            


class LinearDecayGreedyEpsilonPolicy(Policy):
    """Policy with a parameter that decays linearly.

    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.

    - Parameters
      - start_value: float
        The initial value of the parameter
      - end_value: float
        The value of the policy at the end of the decay.
      - decay: int
        The number of steps over which to decay the value.
      - eps: float
        Current epsilon
      - num_actions: int
        The number of actions
    """

    def __init__(self, start_value, end_value, decay, num_actions):  
        self.start_value = start_value
        self.end_value = end_value
        self.decay = float(start_value - end_value) / decay
        self.eps = start_value
        self.num_actions = num_actions
        

    def select_action(self, q_values, **kwargs):
        """Decay parameter and select action.

        - Input
          - q_values: np.array
            The Q-values for each action.
          - is_training: bool, optional
            If true then parameter will be decayed. Defaults to true.

        - Output
          - int
            Selected action index.
        """
        is_training = kwargs.pop('is_training', True)
        x = random.uniform(0, 1) 
        if x > self.eps:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, self.num_actions)
        if is_training:
            self.eps = max(self.eps - self.decay, self.end_value)

        return action

    def reset(self):
        """Start the decay over at the start value."""
        self.eps = start_value
        self.num_steps = 0
