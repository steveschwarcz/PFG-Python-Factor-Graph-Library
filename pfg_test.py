import unittest

import numpy as np
import numpy.testing as npt

from pfg.factor_graph import Variable, Factor, FactorGraph


class TestPFG(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPFG, self).__init__(*args, **kwargs)

    def test_fg_reset(self):
        """
        Test the ability to reset the factor graph.
        """
        student_graph = self.build_student_bayes()

        initial_results = student_graph.posterior_for_all_variables()

        for i in range(5):
            student_graph.belief_propagation_iteration()

        student_graph.reset_state()

        npt.assert_allclose(student_graph.posterior_for_all_variables()['grade'],
                            initial_results['grade'], 1e-4)

    def test_bp_iterative(self):
        """
        Test iterative BP.
        """
        student_graph = self.build_student_bayes()

        for i in range(5):
            student_graph.belief_propagation_iteration()

        inference_results = student_graph.posterior_for_all_variables()

        self.assert_student_inference_results_correct(inference_results)

    def test_bp_tree(self):
        """
        Test BP on a tree graph.
        """
        student_graph = self.build_student_bayes()

        student_graph.belief_propagation_tree()

        inference_results = student_graph.posterior_for_all_variables()

        self.assert_student_inference_results_correct(inference_results)

    def assert_student_inference_results_correct(self, inference_results):
        """
        Assert that given inference results are correct.

        :param inference_results: The results to test.
        """
        npt.assert_allclose(inference_results['grade'], np.array([0.362, 0.2884, 0.3496]), 1e-4)
        npt.assert_allclose(inference_results['letter'], np.array([0.4977, 0.5023]), 1e-4)
        npt.assert_allclose(inference_results['sat'], np.array([0.725, 0.275]), 1e-4)
        npt.assert_allclose(inference_results['diff'], np.array([0.6, 0.4]), 1e-4)
        npt.assert_allclose(inference_results['intel'], np.array([0.7, 0.3]), 1e-4)

    def build_student_bayes(self):
        """
        Builds the standard graph to test against. The values are taken from here:

        https://raw.githubusercontent.com/pgmpy/pgmpy_notebook/master/images/2/student_full_param.png

        :return: The factor graph of the network.
        """
        var_grade = Variable('grade', 3)
        var_diff = Variable('diff', 2)
        var_intel = Variable('intel', 2)
        var_sat = Variable('sat', 2)
        var_letter = Variable('letter', 2)

        factor_g_di_values = np.zeros([3, 2, 2])

        factor_g_di_values[:, 0, 0] = np.array([0.3, 0.4, 0.3])
        factor_g_di_values[:, 1, 0] = np.array([0.05, 0.25, 0.7])
        factor_g_di_values[:, 0, 1] = np.array([0.9, 0.08, 0.02])
        factor_g_di_values[:, 1, 1] = np.array([0.5, 0.3, 0.2])

        factor_d = Factor(np.array([0.6, 0.4]), 'p(diff)')
        factor_i = Factor(np.array([0.7, 0.3]), 'p(intel)')

        factor_g_di = Factor(factor_g_di_values, 'p(grade|diff, intel)')

        factor_l_g = Factor(np.array([[0.1, 0.9], [0.4, 0.6], [0.99, 0.01]]).T, 'p(letter|grades)')
        factor_s_i = Factor(np.array([[0.95, 0.05], [0.2, 0.8]]).T, 'p(sat|intel)')

        graph = FactorGraph()

        graph.add_factor([var_diff], factor_d)
        graph.add_factor([var_intel], factor_i)
        graph.add_factor([var_grade, var_diff, var_intel], factor_g_di)
        graph.add_factor([var_letter, var_grade], factor_l_g)
        graph.add_factor([var_sat, var_intel], factor_s_i)

        return graph


if __name__ == '__main__':
    unittest.main()
