"""
The brain is an interface to give simulation clients a rule set to simulate the
behavior of the drone. A brain consists of input function, that returns a 
behavior as an output.
"""

from abc import ABC
from typing import Callable
import random
from types import FunctionType


class Node(ABC):
    pass


class NonTerminalNode(Node):
    """Class representing a terminal node (leaf) of a tree"""
    def __init__(self, left: Node, right: Node, rule: Callable) -> None:
        """_summary_

        Args:
            left (Node): Left node (false statement)
            right (Node): Right node (true statement)
            rule (Callable): if true, go right
        """
        self.left = left
        self.right = right
        self.rule = rule

    def make_decision(self, input_values):
        """Method to return the child depending on the truth value of the rule

        Args:
            input_values (): values to be processed 

        Returns:
            : next node. Might be a terminal or non terminal node
        """
        return self.left if self.rule(input_values) else self.right


class TerminalNode(Node):
    """Terminal node class. Only contains a single value (vector)"""
    def __init__(self, values: tuple) -> None:
        """Node that holds a vector of specific values

        Args:
            values (tuple): values to be returned
        """
        self.values = values

    def get_values(self) -> tuple:
        """Method to return the value the node represents

        Returns:
            tuple: response value
        """
        return self.values


class Brain(ABC):
    """
    Class that represents the decision making. Classes inheriting from this
    class must implement the process function, that makes a decision for the given input.
    """
    def __init__(self):
        pass

    def process(self, input_value):
        raise NotImplementedError


class DecisionTreeBrain(Brain):
    """A brain that works based on a given decision tree."""
    def __init__(self, decision_tree: NonTerminalNode):
        self.decision_tree = decision_tree

    def process(self, input_value: tuple) -> tuple:
        """Returns the response of the brain to a given state

        Args:
            input_value (tuple): input values to be processed

        Returns:
            tuple: the response of the brain 
        """
        step = self.decision_tree.make_decision(input_value)
        while not isinstance(step, TerminalNode):
            step = step.make_decision(input_value)
        return step.values


class ValueChoice(ABC):
    def __init__(self, choices) -> None:
        self.choices = choices
    def get_random_value(self):
        """Interface to represent a value range that is either
        numerical or categorical"""
        raise NotImplementedError
    

class CategoricalChoices(ValueChoice):
    def __init__(self, choices: list) -> None:
        super().__init__(choices)
    
    def get_random_value(self):
        return random.choice(self.choices)


class NumericalChoices(ValueChoice):
    def __init__(self, choices: list[float, float]) -> None:
        super().__init__(choices)
    
    def get_random_value(self):
        return random.uniform(self.choices[0], self.choices[1])


class VariableChoices(CategoricalChoices):
    """Class for variables in a Parse tree. The order
    of the choices dictates the mapping order when variables
    are replaced by values.
    """
    def get_var_index(self, choice):
        return self.choices.index(choice)
        

class ParseTree:
    def __init__(self, terminal_choices: list[ValueChoice], variable_choices: list[VariableChoices], non_terminal_choices: list[Callable], init_max_length: int) -> None:
        self.terminal_choices = terminal_choices
        self.terminal_choices.append(variable_choices)
        self.variable_choices = variable_choices
        self.non_terminal_choices = non_terminal_choices
        # get random parse tree with max length init_max_length
        self.root = NonTerminalNode(
            self.build_sub_tree(init_max_length), 
            self.build_sub_tree(init_max_length),
            rule=self.get_non_terminal_choice()
        )
    
    def get_non_terminal_choice(self):
        return random.choice(self.non_terminal_choices)

    def get_terminal_choice(self):
        return random.choice(self.terminal_choices).get_random_value()

    def get_random_node(self):
        if random.choice(["T", "N"]) == "N":
            return self.get_non_terminal_choice()
        else:
            return self.get_terminal_choice()
    
    def build_sub_tree(self, max_length: int):
        """
        Builds a random tree recursively to a maximal length of max_length.
        """
        if max_length == 1:
            return TerminalNode(self.get_terminal_choice())
        else:
            new_value = self.get_random_node()
            if isinstance(new_value, FunctionType):
                return NonTerminalNode(
                    self.build_sub_tree(max_length - 1),
                    self.build_sub_tree(max_length - 1),
                    rule=new_value
                )
            else:
                return TerminalNode(new_value)
    
    def get_value(self, variable_values: list) -> float:
        self.variable_values = variable_values
        return self.root.rule(
            self.evaluate(self.root.left),
            self.evaluate(self.root.right)
        )

    def evaluate(self, tree) -> float:
        """Recursive evaluation of the parse tree.
        Variables will be replaced by the variable_values, in order
        
        Args:
            variable_values (list): values that replace the variables, must be same size or at least as long as the number of vars
        
        Returns:
            float: Resulting value of the parsed parse tree"""
        if isinstance(tree, TerminalNode):
            if isinstance(tree.values, str):
                i = self.variable_choices.get_var_index(tree.values)
                return self.variable_values[i]
            return tree.values
        # tree is a Non-Terminal Node
        return tree.rule(
            self.evaluate(tree.left),
            self.evaluate(tree.right)
        )


class ParseTreeBrain(Brain):
    def __init__(self, xy_controller):
        self.xy_controller = xy_controller
    
    def process(self, input_value):
        return self.xy_controller(*input_value), self.xy_controller(*input_value[-2:], *input_value[0:-2])
        
# old pt brain
"""class ParseTreeBrain(Brain):
    def __init__(self, terminals, variables, non_terminals, initial_max_length):
        self.pt_x = ParseTree(terminals, variables, non_terminals, initial_max_length)
        self.pt_y = ParseTree(terminals, variables, non_terminals, initial_max_length)

    def process(self, input_value) -> tuple:
        Returns the response of the brain to a given state.

        
        Args:
            input_value (): input values
        Return:
            tuple: direction to go in
        
        direction = []
        for pt in [self.pt_x, self.pt_y]:
            # evaluate parse tree
            direction.append(pt.get_value(input_value))
        return direction"""

        
