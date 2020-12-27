from abc import abstractmethod

class BaseMAB:
    @abstractmethod
    def chooseArmToPlay(self):
        pass

    @abstractmethod
    def receiveReward(self,arm,reward):
        pass

    def __repr__(self):
        return str(self.__class__.__name__)
    
    def __str__(self):
        return self.__repr__()