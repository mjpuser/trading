import trading.lib.tree
import itertools
import random
import trading.rl

# index to dimension
COL = {
    'change': 0,
    'bollinger': 1,
    'owns': 2,
    'return': 3,
    'action': 4,
}

# index for action
ACTION = {
    'promise': 'promise',
    'buy': 'buy',
    'wait': 'wait',
    'sell': 'sell',
}

DAYS_OWNED_SIZE = 2

# index for action
DACTION = {
    'promise': 0,
    'buy': 1,
    'sell': 2,
    'wait': 3,
}

revmap = {
    '0': 'promise',
    '1': 'buy',
    '2': 'sell',
    '3': 'wait'
}


discrete_changes = [
    -0.013705409864274043,
    -0.007994551909556669,
    -0.004127470678394696,
    -0.001805747215627207,
    0.00038355921955979255,
    0.0028797863193601447,
    0.005756997975481681,
    0.009824218198390078,
    0.015323150458234291,
]

def discretize_change(change):
    for i, bound in enumerate(discrete_changes):
        if change < bound:
            return i
    return len(discrete_changes)


def discretize_return(change):
    return discretize_change(change)
    if -0.005 < change < 0:
        x = 11
    elif 0 <= change < 0.005:
        x = 0

    elif 0.005 <= change < 0.015:
        x = 6
    elif 0.015 <= change < 0.025:
        x = 7
    elif 0.025 <= change < 0.035:
        x = 8
    elif 0.035 <= change < 0.045:
        x = 9
    elif 0.045 <= change:
        x = 10

    elif -0.015 < change <= -0.005:
        x = 1
    elif -0.025 < change <= -0.015:
        x = 2
    elif -0.035 < change <= -0.025:
        x = 3
    elif -0.045 < change <= -0.035:
        x = 4
    elif change <= -0.045:
        x = 5
    return x

def discretize_action(action):
    # return None in case invalid action
    return action

def discretize_owns(days):
    return min(days, DAYS_OWNED_SIZE - 1)
# states is % change, bollinger, owns, return, action
# b - - - s b - - - s
# - b - - - s b - - - s
# - - b - - - s b - - - s
# - - - b - - - s b - - - s
# - - - - b - - - s b - - - s
# b - - - s - - - b
def state_generator(states, learner, randomness=1):
    ret = 0
    days_owned = 0
    for state in states:
        change, bollinger = state
        if days_owned > 0:
            ret = (1 + ret) * (1 + change) - 1
        prefix = (change, bollinger, days_owned, ret,)
        allowed_actions = actions_filter(prefix)
        if random.random() < randomness:
            action = random.choice(allowed_actions)
        else:
            *_, action = learner.argmax(learner.discretize((change, bollinger, days_owned, ret, None)))
        yield change, bollinger, days_owned, ret, action
        # print('----------------')
        # print('change', change)
        # print('bollinger', bollinger)
        # print('owned', days_owned)
        # print('return', ret)
        # print('action', revmap[str(action)], allowed_actions)
        if action == DACTION['buy']:
            days_owned = 1
            ret = 0
        elif action == DACTION['sell']:
            days_owned = 0
        elif action == DACTION['promise'] and days_owned < DAYS_OWNED_SIZE - 1:
            days_owned += 1

def reward(state):
    ret = 0
    action = state[COL['action']]
    if action == DACTION['sell']:
        ret = state[COL['return']]
    return ret

def actions_filter(state):
    if state[COL['owns']] > 0:
        return (DACTION['sell'], DACTION['promise'],)
    else:
        return (DACTION['wait'], DACTION['buy'],)

def discretize(state):
    change, bollinger, owns, ret, action = state
    return (
        discretize_change(change),
        bollinger,
        discretize_owns(owns),
        discretize_return(ret),
        discretize_action(action),
    )


class Learner(trading.rl.Q):

    def __init__(self, table=None):
        shape = (11, 3, DAYS_OWNED_SIZE, 12, 4,)
        super(Learner, self).__init__(
            reward,
            shape,
            discretize,
            actions_filter,
            table
        )

    def predict(self, states, output=False):
        total_return = 0
        ret = 0
        owns = 0
        for change, bollinger in states:
            if owns > 0:
                ret = (1 + ret) * (1 + change) - 1
            *_, action = self.argmax(self.discretize((change, bollinger, owns, ret, None,)))
            if output:
                print('discretized', _)
            b = 'mid'
            if bollinger == 1:
                b = 'abv'
            elif bollinger == 2:
                b = 'bel'
            if action == DACTION['buy']:
                owns = 1
                ret = 0
            elif action == DACTION['sell']:
                owns = 0
                total_return += ret
                ret = 0
            elif action == DACTION['promise'] and owns < DAYS_OWNED_SIZE - 1:
                owns += 1
            *disc, _ = self.discretize((change, bollinger, owns, ret, None,))
            if output:
                print((b, owns, change, ret, revmap[str(action)],))

        # print('total_return', total_return)
        return total_return, ret
