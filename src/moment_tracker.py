import numpy as np

class MomentTracker:
    """A convenience class that tracks the first two moments (the mean and standard deviation) independently in a number of dimensions for a set of inputs over time-steps. It can also be used to standardize inputs.
    """

    SMALL_POSITIVE_NUMBER = 1.0e-9

    def __init__(self, name, dim, verbose_at=0):
        """Initializes an instance of the class.
        
        Params
        ======
        name (str): The name of the tracker.
        dim (int): The dimensionality of the inputs. Each dimension is treated univariately.
        verbose_at (int): The number of steps at which to output summary statistics. 0 for never.
        """
        self.name = name
        self.dim = dim
        self.verbose_at = verbose_at
        self.previous_verbose = 0

        self.reset()

    def reset(self):
        """Resets the tracker.
        """
        self.counter = 0
        self.m1 = np.zeros((self.dim))
        self.m2 = np.zeros((self.dim))
        
    def update(self, input_):
        """Updates the tracker based on some input.
        
        Params
        ======
        input_: The input with which to update the tracker.
        """
        assert input_.shape[1] == self.dim

        self.counter += input_.shape[0]
        self.m1 += np.sum(input_, axis=0)
        self.m2 += np.sum(input_**2, axis=0)

        if self.verbose_at > 0 and self.counter - self.previous_verbose >= self.verbose_at:
            print(f'{self.name}: {self.counter}: {self.get_moments()}')
            self.previous_verbose = self.counter

    def get_moments(self):
        """Returns the moments of the tracker."""        
        means = self.m1 / self.counter
        variances = np.maximum(self.m2 / self.counter - (means)**2, 0)        
        stds = np.sqrt(variances)
        return (means, stds)

    def apply(self, input_):
        """Uses the tracker's moments to standardize some input.
        
        Params
        ======
        input_: The input to standardise.
        """
        means, stds = self.get_moments()
        result = (input_ - means) / (stds + MomentTracker.SMALL_POSITIVE_NUMBER)        
        return result

    
    def update_and_apply(self, input_):
        """Updates the tracker based on some input and then standardizes that input.
        
        Params
        ======
        input_: The input to apply and then standardize.
        """
        self.update(input_)
        return self.apply(input_)
