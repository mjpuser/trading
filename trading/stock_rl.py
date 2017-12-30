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
    'promise': 0,
    'buy': 1,
    'sell': 2,
    'wait': 3,
}

BOLLINGER = {
    'inside': 0, # inside
    'below': 1, # below the band
    'dipped': 2, # went from below to inside
    'above': 3, # above the band
    'returned': 4, # went from above to inside
}

revmap = list(ACTION.keys())

def discretize_action(action):
    # return None in case invalid action
    return ACTION[action]

def discretize_owns(owns):
    return owns

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
    # change, bollinger, owns, buystate, return, action
    _, bollinger, owns, buystate, _, action = state
    return (
        BOLLINGER[bollinger],
        discretize_owns(owns),
        BOLLINGER[buystate],
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
