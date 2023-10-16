"""
The brain is an interface to give simulation clients a rule set to simulate the
behavior of the drone. A brain consists of input function, that returns a 
behavior as an output.
"""

from abc import ABC


class Brain(ABC):
    """
    Class that represents the decision making. Classes inheriting from this
    class must implement the process function, that makes a decision for the given input.
    """
    def __init__(self):
        pass

    def process(self, input_value):
        raise NotImplementedError


class ParseTreeBrain(Brain):
    def __init__(self, xy_controller):
        self.xy_controller = xy_controller
    
    def process(self, input_value):
        return self.xy_controller(*input_value), self.xy_controller(*input_value[-2:], *input_value[0:-2])
