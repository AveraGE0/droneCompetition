from simulation.brain import TerminalNode, NonTerminalNode, ParseTree, NumericalChoices, CategoricalChoices, VariableChoices


def test_parse_tree():
    pt = ParseTree(
        terminal_choices=[NumericalChoices([0, 10])],
        variable_choices=VariableChoices(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        non_terminal_choices=[lambda x, y: x + y, lambda x, y: x * y, lambda x, y: x / y, lambda x, y: x - y],
        init_max_length=3
    )
    print(pt.get_value([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
    pt_add =ParseTree([NumericalChoices([1, 2])], VariableChoices(["x", "y", "z"]), [lambda x, y: x + y], init_max_length=1)
    pt_add.root = NonTerminalNode(
        left=NonTerminalNode(
            left=TerminalNode("x"),
            right=TerminalNode("z"),
            rule=lambda x, y: x * y
        ),
        right=TerminalNode("y"),
        rule=lambda x, y: x + y
    )
    assert pt_add.get_value([4, 2, 4]) == 18, "The result should be 18"

    

if __name__ == '__main__':
    test_parse_tree()