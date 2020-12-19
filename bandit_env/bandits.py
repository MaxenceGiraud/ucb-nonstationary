import numpy as np

class MAB:
    def __init__(self,arms):
        """given a list of arms, create the MAB environnement"""
        self.arms = arms
        self.nbArms = len(arms)
        self.means = [arm.mean for arm in arms]
        self.bestarm = np.argmax(self.means)
    
    def generateReward(self,arm):
        return self.arms[arm].sample()

class MAB_NS(MAB):
    ''' Non stationary Bandit 
    
    Parameters
    ----------
    n_arms : int,
        Number of arms of the bandit 
    arms_list : list,
        list of arms object from which the arms are drawn
    p : float,
        Probability of an arm to change at each step
    init_arms : list,
        Initialization of arms list. If None choose at random. Defaults to None.
    '''
    def __init__(self,n_arms,arms_list,p=0.01,init_arms = None):
        self.nbArms = n_arms
        if init_arms is None :
            self.arms =arms_list[np.random.randint(len(arms_list))[:n_arms]]
        else : 
            self.arms = init_arms
        self.arms_list = arms_list
        self.p  = p
    
    @property
    def means(self):
        return [arm.mean for arm in self.arms]
    
    @property
    def bestarm(self):
        return np.argmax(self.means)
    
    def generateReward(self,arm):
        # Randomly change arms with proba p 
        changing_arms = np.where(np.random.random(size=self.nbArms) < self.p)
        self.arms[changing_arms] = self.arms_list[np.random.randint(len(self.arms_list))[:len(changing_arms)]]
        
        return super().generateReward(arm)