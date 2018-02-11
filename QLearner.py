"""Implement QLearner, a Reinforcement Learning class"""

import numpy as np
import random as rand
from copy import deepcopy

class QLearner(object):

    def __init__(self, num_states=100, num_actions=4, alpha=0.2,
        gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
        """The constructor QLearner() reserves space for keeping track of Q[s, a] for 
        the number of states and actions. It initializes Q[] with all zeros.

        Parameters:
        num_states: int, the number of states to consider
        num_actions: int, the number of actions available
        alpha: float, the learning rate used in the update rule. 
               Should range between 0.0 and 1.0 with 0.2 as a typical value
        gamma: float, the discount rate used in the update rule. 
               Should range between 0.0 and 1.0 with 0.9 as a typical value.
        rar: float, random action rate. The probability of selecting a random action 
             at each step. Should range between 0.0 (no random actions) to 1.0 
             (always random action) with 0.5 as a typical value.
        radr: float, random action decay rate, after each update, rar = rar * radr. 
              Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
        dyna: int, conduct this number of dyna updates for each regular update. 
              When Dyna is used, 200 is a typical value.
        verbose: boolean, if True, your class is allowed to print debugging 
                 statements, if False, all printing is prohibited.
        """        
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        # Keep track of the latest state and action which are initialized to 0
        self.s = 0
        self.a = 0
        
        # Initialize a Q table which records and updates Q value for
        # each action in each state
        self.Q = np.zeros(shape=(num_states, num_actions))
        # Keep track of the number of transitions from s to s_prime for when taking 
        # an action a when doing Dyna-Q
        self.T = {}
        # Keep track of reward for each action in each state when doing Dyna-Q
        self.R = np.zeros(shape=(num_states, num_actions))

    def query_set_state(self, s):
        """Find the next action to take in state s. Update the latest state and action 
        without updating the Q table. Two main uses for this method: 1) To set the  
        initial state, and 2) when using a learned policy, but not updating it.

        Parameters:
        s: The new state
        
        Returns: The selected action to take in s
        """
        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = self.Q[s, :].argmax()
        self.s = s
        self.a = action
        if self.verbose: 
            print ("s =", s,"a =",action)
        return action

