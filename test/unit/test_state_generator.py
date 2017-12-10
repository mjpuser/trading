import unittest
import trading.stock_state

class StateGeneratorTestCase(unittest.TestCase):

    def test_paths(self):
        paths = [
            path
            for path in trading.stock_state.state_generator(horizon=5)
        ]
        expected = [
            ('buy', 'sell',),
            ('buy', None, 'sell',),
            ('buy', None, None, 'sell',),
            ('buy', None, None, None, 'sell',),
        ]
        self.assertEqual(expected, paths)
