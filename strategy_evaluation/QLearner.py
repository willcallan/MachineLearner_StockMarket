""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: William Callan  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT User ID: wcallan3  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT ID: 903546349  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import random as rand  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class QLearner(object):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    This is a Q learner object.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param num_states: The number of states to consider.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type num_states: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param num_actions: The number of actions available..  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type num_actions: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type alpha: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type gamma: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type rar: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type radr: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type dyna: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type verbose: bool  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def __init__(  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        num_states=100,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        num_actions=4,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        alpha=0.2,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        gamma=0.9,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        rar=0.5,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        radr=0.99,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        dyna=0,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        verbose=False,
    ):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Constructor method  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        # save all the class variables
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        # populate the Q-table and set the starting action / state
        self.table = np.zeros((num_states, num_actions))
        #self.table = np.random.randn(num_states, num_actions) / 10000
        self.s = 0
        self.a = 0
        # initialize the dyna modelled tables: T[s, a, s'] and R[s, a]
        self.past_experiences = np.empty((0, 3), dtype=np.int_)
        self.model_r = np.zeros((num_states, num_actions))

    def author(self):
        return "wcallan3"
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def querysetstate(self, s):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Update the state without updating the Q-table  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param s: The new state  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type s: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :return: The selected action  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :rtype: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        # determine if a random action should be chosen
        if rand.uniform(0, 1) <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        # otherwise choose the action in s's row with the highest value
        else:
            action = np.argmax(self.table, axis=1)[s]

        # update the previous state / action
        self.s = s
        self.a = action

        # print data
        if self.verbose:
            print(f"s = {s}, a = {action}")

        return action

    def query(self, s_prime, r):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Update the Q table and return an action  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param s_prime: The new state  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type s_prime: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param r: The immediate reward  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type r: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :return: The selected action  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :rtype: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        # update the Q-table for the previous state / action
        improved_est = r + self.gamma * np.max(self.table, axis=1)[s_prime]
        self.table[self.s, self.a] = (1 - self.alpha) * self.table[self.s, self.a] + self.alpha * improved_est

        # ----Dyna-Q----
        if self.dyna > 0:
            # update models
            self.past_experiences = np.append(self.past_experiences, [np.array([self.s, self.a, s_prime])], axis=0)
            self.model_r[self.s, self.a] = (1 - self.alpha) * self.model_r[self.s, self.a] + self.alpha * r
            # hallucinate _dyna_ times
            hallucinations = self.past_experiences[np.random.choice(self.past_experiences.shape[0], size=self.dyna)]
            hallucinated_r_vals = self.model_r[hallucinations[:, 0], hallucinations[:, 1]]
            # update Q-table
            for i in range(self.dyna):
                improved_est = hallucinated_r_vals[i] + self.gamma * np.max(self.table, axis=1)[hallucinations[i, 2]]
                self.table[hallucinations[i, 0], hallucinations[i, 1]] = (1 - self.alpha) * self.table[hallucinations[i, 0], hallucinations[i, 1]] + self.alpha * improved_est
        # ----Dyna-Q----

        # determine if a random action should be chosen
        if rand.uniform(0, 1) <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        # otherwise choose the action in s_prime's row with the highest value
        else:
            action = np.argmax(self.table, axis=1)[s_prime]
        # decay randomness
        self.rar = self.rar * self.radr

        # update the previous state / action
        self.s = s_prime
        self.a = action

        # print data
        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        return action
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("Remember Q from Star Trek? Well, this isn't him")
