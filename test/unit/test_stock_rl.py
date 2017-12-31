import unittest
import trading.stock_rl
import test.unit.data

BOLLINGER = trading.stock_rl.BOLLINGER
ACTION = trading.stock_rl.ACTION

class StateGeneratorTestCase(unittest.TestCase):
    def test_discretizer(self):
        # takes change, bollinger, owns, buystate, return, action
        # returns to bollinger, owns, buystate, action
        state = (
            0.01,
            'inside',
            True,
            'above',
            0.10,
            'buy',
        )

        dstate = trading.stock_rl.discretize(state)

        expected = (
            BOLLINGER['inside'],
            True,
            BOLLINGER['above'],
            ACTION['buy'],
        )

        self.assertEqual(dstate, expected)

    def test_actions_filter_owns(self):
        owns = True
        state = (
            0.01,
            'inside',
            owns,
            None,
            0,
            None,
        )
        actions = trading.stock_rl.actions_filter(state)
        expected = (ACTION['sell'], ACTION['promise'],)
        self.assertEqual(actions, expected)

    def test_actions_filter_no_shares(self):
        owns = False
        state = (
            0.01,
            'inside',
            owns,
            None,
            0,
            None,
        )
        actions = trading.stock_rl.actions_filter(state)
        expected = (ACTION['wait'], ACTION['buy'],)
        self.assertEqual(actions, expected)

    def test_state_generator(self):
        # states are change, bollinger
        states = [
            (0.01, 'inside',),
        ]
        learner = trading.stock_rl.Learner()
        # full state: change, bollinger, owns, buystate, return, action
        expected = (
            0.01,
            'inside',
            False,
            None,
            0,
            None,
        )
        for state in trading.stock_rl.state_generator(states, learner, 0):
            *expected, action = expected
            *state, action = state
            self.assertEqual(state, expected)
