import numpy as np

class RandomStateMixin:
    """
    Mixin to set random state
    """
    def set_random_state(self, random_state):
        """
        sets the random_state member of the object
        
        Args:
            random_state (int/np.random.RandomState/None): the random state initializer
        """
        
        self._random_state_init= random_state
        
        if random_state is None:
            self.random_state= np.random
        elif isinstance(random_state, int):
            self.random_state= np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state= random_state
        elif random_state is np.random:
            self.random_state= random_state
        else:
            raise ValueError("random state cannot be initialized by " + str(random_state))
    