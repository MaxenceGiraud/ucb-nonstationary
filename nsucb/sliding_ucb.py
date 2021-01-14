from .ucb import UCB
from .randmax import randmax
import numpy as np

class SlidingUCB(UCB):
    """
    Sliding-window UCB algorithm.

    Parameters
    ----------
    nbArms : int
        Number of arms of the considered bandit.
    tau : int
        Number of last time steps to consider to calculate the mean.
    B : float
        An upper bound of the rewards.
        The larger B, the larger the padding function.
    xi : float
        A positive constant for the padding function.
        The larger xi, the larger the padding function.

    Returns
    -------
    None.

    """
    
    def __init__(self, nbArms, tau, B, xi):
        assert xi > 0, "Xi must be positive."
        self.nbArms = nbArms
        self.tau = tau
        self.B = B
        self.xi = xi
        self.clear()
        
        
    def clear(self):
        # history of played arms, of size t
        self.history = []
        
        # history of played arms as booleans, of size (t,nbArms)
        self.history_bool = None 
        
        # successive rewards, of size (t, nbArms), to keep track of them
        # in order to compute the sum X_t(tau, i), denoted by X here
        self.rewards = None
        super().clear()
        
        
    def chooseArmToPlay(self):
        if self.t < self.nbArms:
            return self.t
        else:
            N = np.sum(self.history_bool[-self.tau:], axis=0)
            
            X = (1/N) * np.sum(self.rewards[-self.tau:], axis=0)
            c = self.B * np.sqrt((self.xi * np.log(max(self.t,
                                                       self.tau)))/N)
            
            # N can contain 0s, which leads to nan in X.
            # Then the value of X+c should be +inf, because the sum
            # in X vanishes. Doing so also makes the corresponding arm
            # to be chosen automatically, because for all i,
            # X_t(gamma, i) >= 0 and N_t(gamma, i) >= 0.
            # This is what this line does:
            bound = np.nan_to_num(X+c, copy=False, nan=np.inf)
            return randmax(bound)
            
            
    def receiveReward(self, arm, reward):
        # add to history
        self.history.append(arm)
        
        # add to history_bool
        arm_bool = np.zeros(self.nbArms)
        arm_bool[arm] = 1
        if self.history_bool is None : 
            # this is for t=0
            # it is only a trick for initialization and then vstack
            self.history_bool = arm_bool
        else : 
            self.history_bool = np.vstack((self.history_bool,
                                           arm_bool))
        
        # add reward to self.rewards
        # same trick for the rewards
        reward_this_step = np.zeros(self.nbArms)
        reward_this_step[arm] = reward
        if self.rewards is None:
            # first step, t=0
            self.rewards = reward_this_step
        else:
            self.rewards = np.vstack((self.rewards,
                                     reward_this_step))
        super().receiveReward(arm,reward)
        

class SlidingUCB_uniform_discount(SlidingUCB):
    '''Variant Of Sliding UCB with discount factor added in the N '''

    def chooseArmToPlay(self):
        if self.t < self.nbArms:
            return self.t
        else:
            N = np.sum(self.history_bool[-self.tau:], axis=0) +  np.sum(self.history_bool[:-self.tau], axis=0) / self.tau

            X = (1/N) * np.sum(self.rewards[-self.tau:], axis=0)
            c = self.B * np.sqrt((self.xi * np.log(max(self.t,
                                                       self.tau)))/N)

            return randmax(X+c)


if __name__ == "__main__":
    my_SWUCB = SlidingUCB(4, 6, 2, 3)
    print(my_SWUCB)