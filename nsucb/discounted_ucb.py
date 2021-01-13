from .randmax import randmax
import numpy as np
from .ucb import UCB

class DiscountedUCB(UCB):
    ''' Discounted UCB algorithm

    Parameters 
    ----------
    nbArms :int,
        Number of arms of bandit
    gamma : float,
        Discount factor. Must be between 0 and 1
    B : float,
        upper bound  on all means
    Xi : float,
        Some appropritate constante that scale the discounted padding function
    '''
    def __init__(self,nbArms,gamma=0.99,B=1,xi=1):
        assert gamma <=1 and gamma >= 0, "Gamma must be in the [0,1] interval"
        self.nbArms = nbArms
        self.gamma = gamma
        self.B = B
        self.xi = xi
        self.clear()
    
    def clear(self):
        self.history = [] # history of played arms, of size t
        self.history_bool = None # # history of played arms as booleans, of size (t,nbArms)
        self.reward_history = []
        super().clear()

    def chooseArmToPlay(self):
        if self.t < self.nbArms :
            return self.t
        else : 
            discount = np.ones(self.t)*self.gamma**(self.t-np.arange(self.t))
            N = np.array([np.sum(discount[np.where(i==self.history,1,0)]) for i in range(self.nbArms)])
            print("N for D-UCB=", N)
            X = 1/N * np.sum(discount.reshape(-1,1) *np.reshape(self.reward_history,(-1,1))* self.history_bool,axis=0) # discounted empirical average

            c = 2 * self.B * np.sqrt((self.xi * np.log(N.sum())/N))# discounted padding function   

            return randmax(X+c)
        
    def receiveReward(self,arm,reward):
        self.history.append(arm)
        self.reward_history.append(reward)
        arm_bool = np.zeros(self.nbArms)
        arm_bool[arm] = 1
        
        if self.history_bool is None : 
            self.history_bool = arm_bool
        else : 
            self.history_bool = np.vstack((self.history_bool,arm_bool))
        super().receiveReward(arm,reward)