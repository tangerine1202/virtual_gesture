from abc import ABC, abstractmethod


class PosVelFilter(ABC):
    @abstractmethod
    def __init__(self):
        """
        Initialize the configuration of the filter.
        """
        pass

    @abstractmethod
    def build(self, z=None):
        """
        Build the filter.

        Parameters
        ---
        z: np.array(), None
        The first measurement.
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Update the estimate without measurement.
        """
        pass

    @abstractmethod
    def update(self, z):
        """
        Update the estimate with measurement.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save the current state of the filter.
        """
        pass

    @property
    @abstractmethod
    def x(self):
        """
        Estimate of the variable.
        """
        pass

    @property
    @abstractmethod
    def dx(self):
        """
        Estimate of the rate of change of the variable.
        """
        pass
