# General utility for tracking that don't fit under the 'linear' or 'nonlinear' umbrella
import numpy as np

class LatticeBase():
    """Provide a general parent class for lattices, where there are methods that might be useful"""

    def __init__(self):
        # Do not think that the below method of creating these attributes is ideal
        self.elements_list: list   = NotImplemented
        self.previous_distances: list = NotImplemented
        self.elements_len: list       = NotImplemented
        self.elements_number: int     = NotImplemented


    def _basic_travel_distance_calculation(self, delta_s):
        """An internal utility function, to find how far the beam will travel through each element.
        Minimum is 0, maximum is the length of the element"""
        s_elements = [np.maximum(np.minimum(delta_s-self.previous_distances[i], self.elements_len[i]), 0.0)
                      for i in range(self.elements_number)]
        return s_elements
