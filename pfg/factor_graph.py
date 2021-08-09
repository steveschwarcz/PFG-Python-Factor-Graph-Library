import pprint

import numpy as np


class Variable:
    """
    Encapsulates a variable in a factor graph.
    """
    def __init__(self, name, dim):
        """
        Create a variable.

        :param name: The name of the variable.
        :param dim: The number of states the variable can be in.
        """
        self.name = name
        self.dim = dim

    def __repr__(self):
        return '(' + self.name + ', ' + repr(self.dim) + ')'


class Factor:
    """
    Encapsulates a factor in a factor graph.
    """
    def __init__(self, values, name=None, category=None):
        """
        Create a factor for the factor graph.

        :param values: The values of the factor graph.  Should be a tensor of shape matching the dimensions of the
            variables.  i.e. N1 x N2 x ... x Nd
        :param name: Optional name to give to this factor.
        :param category: Optional factor category object.
        """
        self.values = values
        self.name = name
        self.category = category

    def __repr__(self):
        prefix = '{' + self.name + ', ' if self.name is not None else '{'

        return prefix + repr(self.values.shape) + '}'


class FactorCategory:
    """
    Factor categories represent groups of factors.  These can be used to order different types of factors into
    convenient orderings for iterative loopy belief propagation.
    """
    def __init__(self, name):
        """

        :param name: The name of the category.
        """
        self.name = name


class FactorGraph:
    """
    Encapsulates a factor graph.
    """
    EPSILON = 1e-30

    def __init__(self):
        self.variable_edges = {}    # A dictionary of sets that lead from each variable to their factors.
        self.factor_edges = {}  # A dictionary of lists that lead from each factor to its variables.
        self.variables = set()      # All available variables.
        self.factors = set()    # All available factors.

        self.factors_by_category = {}  # Factors indexed by category.

        self._messages = {}

        self._schedule = []               # The schedule to use for belief propagation as a list of factors.

    def add_factor(self, variables, factor):
        """
        Add a factor to the graph.

        :param variables: The variables of the factor.  Will be automatically added to the graph if they are not already
            present.
        :param factor: The factor of the factor graph.
        :return: The Factor object that was added to the graph.
        """
        # Sanity check on variable dimensions.
        for idx, variable in enumerate(variables):
            if variable.dim != factor.values.shape[idx]:
                raise ValueError('Dimension of variable %s does not match the dimension of factor %s.' %
                                 (repr(variable), repr(factor)))

        for idx, variable in enumerate(variables):
            if variable not in self.variable_edges:
                self.add_variable(variable)

            self.variable_edges[variable].add(factor)

            self._initialize_message_for_variable(variable, factor, idx)

            # By default we schedule factors in the order they are added.
            self._schedule.append((factor, variable))

        # Possibly add to the appropriate category.
        if factor.category is not None:
            if factor.category not in self.factors_by_category:
                self.factors_by_category[factor.category] = []

            self.factors_by_category[factor.category].append(factor)

        # Add variables to factor edges.
        self.factor_edges[factor] = variables

        self.factors.add(factor)

    def add_variable(self, variable):
        """
        Add a variable to the graph.

        :param variable: The variable to add.
        """
        if variable in self.variables:
            raise ValueError('Variable %s already exists and cannot be added.' % repr(variable))

        self.variables.add(variable)
        self.variable_edges[variable] = set()

    def add_variables_from_list(self, variables):
        """
        Add variables to the graph from a list of variables.

        :param variables: The variables to add.
        """
        for variable in variables:
            self.add_variable(variable)

    def reset_state(self):
        """
        Reset the state of all messages.  If a factor's values have been changed, this will be necessary.
        """
        for factor in self.factors:
            for idx, variable in enumerate(self.factor_edges[factor]):
                self._initialize_message_for_variable(variable, factor, idx)

    def set_schedule(self, schedule):
        """
        Set the schedule for loopy belief propagation.

        :param schedule: A list of factors, variables, or factor categories.
        """
        self._schedule = []
        for elem in schedule:
            if isinstance(elem, Factor):
                for variable in self.factor_edges[elem]:
                    self._schedule.append((elem, variable))
            if isinstance(elem, Variable):
                for factor in self.variable_edges[elem]:
                    self._schedule.append((elem, factor))
            elif isinstance(elem, FactorCategory) and elem in self.factors_by_category:
                for factor in self.factors_by_category[elem]:
                    for variable in self.factor_edges[factor]:
                        self._schedule.append((factor, variable))

    def belief_propagation_iteration(self):
        """
        Perform an iteration of belief propagation.
        """
        for edge in self._schedule:
            self._update_message_edge(edge)

        for edge in reversed(self._schedule):
            self._update_message_edge((edge[1], edge[0]))

    def belief_propagation_tree(self):
        """
        Perform tree belief propagation.
        """
        # Pick an arbitrary root.
        root = self._schedule[0][0]

        # Make a schedule that can guarantee convergence in one iteration.
        self._schedule = self._build_schedule_for_tree(root, None)

        self.belief_propagation_iteration()

    def posterior_for_all_variables(self):
        """
        Perform inference for all variables.

        :return: A dictionary with each variable name (not the variable itself) as a key, and it's vector of inferences
            as a value.
        """
        return {variable.name: self.posterior_for_variable(variable) for variable in self.variables}

    def posterior_for_variable(self, variable):
        """
        Perform inference for a given variable.

        :param variable: The variable to perform inference on.
        :return: A vector of size `variable.dim` containing the final probabilities for the given variable's state.
        """
        result = np.ones([variable.dim])

        for neighbor in self.variable_edges[variable]:
            result *= self._messages[neighbor, variable]

        return result / np.maximum(np.sum(result), self.EPSILON)

    def _build_schedule_for_tree(self, current, parent, already_visited=None):
        """
        Builds a schedule for running BP on a tree graph structure.  Schedule is built using a recursive depth
            first search.

        :param current: The current variable in the search.
        :param parent: The parent of the current variable in the search.  Should be None for the root.
        :param already_visited: A set of variables already visited by the search.  Used to ensure graph is actually a
            tree.
        :return: A schedule of edges going from the leaves to the root.
        """
        # Keep track of already visited variables.
        if already_visited is None:
            already_visited = {current}
        else:
            already_visited.add(current)

        schedule = [(current, parent)] if parent is not None else []

        # Find the children of this variable.
        if isinstance(current, Factor):
            children = self.factor_edges[current]
        else:
            children = self.variable_edges[current]

        # Recursively visit other children.
        for child in children:
            if child != parent:
                if child in already_visited:
                    raise ValueError("Cannot run Tree BP.  Graph is a not a Tree.")

                schedule = self._build_schedule_for_tree(child, current, already_visited) + schedule

        return schedule

    def _update_message_edge(self, edge):
        """
        Pass a message along an edge.

        :param edge: The edge to pass along.  May be a tuple of (Factor, Variable), or (Variable, Factor).
        """
        if isinstance(edge[0], Factor):
            self._update_message_factor_to_variable(edge[0], edge[1])
        else:
            self._update_message_variable_to_factor(edge[0], edge[1])

    def _initialize_message_for_variable(self, variable, factor, variable_idx):
        """
        Initializes the messages for a given variable factor pair.

        :param variable: The variable.
        :param factor: The factor.
        :param variable_idx: The index of the variable with respect to the factor's values.
        """
        self._messages[(variable, factor)] = np.ones([variable.dim])

        # Only sum over evey axis but the one containing this variable's values.
        axes_to_sum = list(range(len(factor.values.shape)))
        axes_to_sum.remove(variable_idx)

        if len(axes_to_sum) == 0:
            self._messages[(factor, variable)] = factor.values
        else:
            self._messages[(factor, variable)] = np.sum(factor.values, axis=tuple(axes_to_sum))

    def _update_message_factor_to_variable(self, from_factor, to_variable):
        """
        Updates a message being passed from a factor to a variable.  Use iteratively with a cyclic structure.

        :param from_factor: The factor from which the message will be coming.
        :param to_variable: The variable to which the message will be going.
        """
        neighbors = list(self.factor_edges[from_factor])
        neighbors.remove(to_variable)

        # Only sum over evey axis but the one containing this variable's values.
        variable_idx = self._variable_idx_for_factor(to_variable, from_factor)
        axes_to_sum = list(range(len(self.factor_edges[from_factor])))
        axes_to_sum.remove(variable_idx)

        if len(neighbors) == 0:
            # Base case.
            return self._messages[(from_factor, to_variable)]
        else:
            # Compute message from incoming messages.
            incoming_messages = [self._messages[(neighbor, from_factor)] for neighbor in neighbors]

            # We combine all messages into a single outer product tensor.
            mult_factor = self._outer_product_multiple(incoming_messages)

            # We expand the dimensions so that they match that of the values array for this factor, except with a
            # 1 for the shape of the variable idx.  This way when we multiply broadcasting takes care of the heavy
            # lifting.
            mult_factor = np.expand_dims(mult_factor, axis=variable_idx)

            term_to_sum = from_factor.values * mult_factor

            message = np.sum(term_to_sum, axis=tuple(axes_to_sum))

            self._messages[(from_factor, to_variable)] = message / np.sum(message, axis=None)

            return message

    def _update_message_variable_to_factor(self, from_variable, to_factor):
        """
        Calculates a message being passed from a factor to a variable.  Use iteratively with a cyclic structure.

        :param from_variable: The variable from which the message will be coming.
        :param to_factor: The factor to which the message will be going.
        """
        neighbors = self.variable_edges[from_variable] - {to_factor}

        if len(neighbors) > 0:
            message = np.ones([from_variable.dim])

            for neighbor in neighbors:
                if np.max(message) == float('inf'):
                    raise ArithmeticError('Encountered a value of infinity during belief propagation.  It may be '
                                          'necessary to ensure all factors sum to 1.')
                message *= self._messages[(neighbor, from_variable)]

            self._messages[(from_variable, to_factor)] = message * 1e20 # + self.EPSILON

        return self._messages[(from_variable, to_factor)]

    def _variable_idx_for_factor(self, variable, factor):
        """
        Gets the index of a variable in a corresponding factor.

        :param variable: The variable.
        :param factor: The factor.
        :return: The index of the variable for the factor.
        """
        return self.factor_edges[factor].index(variable)

    @classmethod
    def _outer_product_multiple(cls, tensors):
        """
        Gets the outer product of all tensors in a list of tensors.

        :param tensors: A list of tensors.
        :return: The outer product of all the tensors.  The shape of the tensor returned is just the shape of all the
            tensors that went into the product concatenated.
        """
        result = tensors[0]

        for i in range(len(tensors) - 1):
            result = cls._outer_product(result, tensors[i + 1])

        return result

    @classmethod
    def _outer_product(cls, a, b):
        """
        Takes the outer product of two tensors, while preserving their shapes.

        :param a: The first tensor.
        :param b: The second tensor.
        :return: The outer product of a and b.  The shape of the last tensor is the shape of a concatenated with the
            shape of b.
        """
        return np.reshape(np.outer(a, b), a.shape + b.shape)

    def __repr__(self):
        return 'Variable Edges: %s\nFactor Edges: %s' % \
               (pprint.pformat(self.variable_edges, width=1), pprint.pformat(self.factor_edges, width=1))


if __name__ == '__main__':
    pass

