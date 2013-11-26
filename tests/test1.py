from unittest import TestCase

from simple_simplex import optimize, LP_STATE

class TestSolver(TestCase):
    def test_1(self):
        LP = """
x_1 - x_2 + x_3
2x_1 - x_2 + 2x_3 <= 4
2x_1 - 3x_2 + x_3 <= -5
-x_1 + x_2 - 2x_3 <= 1
        """.strip()

        status, value, variables = optimize(LP)

        self.assertEquals(status, LP_STATE.OPTIMAL)

        self.assertAlmostEqual(value, 0.6)

        self.assertAlmostEqual(variables['x_1'], 0)
        self.assertAlmostEqual(variables['x_2'], 2.8)
        self.assertAlmostEqual(variables['x_3'], 3.4)

