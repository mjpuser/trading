import trading.lib.tree
import itertools
import random
import trading.rl

# index to dimension
COL = {
    'change': 0,
    'bollinger': 1,
    'owns': 2,
    'buystate': 3,
    'return': 4,
    'action': 5,
}

DCOL = {
    'owns': 1
}
# index for action
ACTION = {
    'promise': 0,
    'buy': 1,
    'sell': 2,
    'wait': 3,
}
RACTION = list(ACTION.keys())

BOLLINGER = {
    'inside': 0, # inside
    'below': 1, # below the band
    'dipped': 2, # went from below to inside
    'above': 3, # above the band
    'returned': 4, # went from above to inside
    'None': 5,
}
RBOLLINGER = list(BOLLINGER.keys())

def discretize_action(action):
    # return None in case invalid action
    return ACTION.get(action)

def discretize_owns(owns):
    return 1 if owns else 0

def state_generator(states, learner, randomness=1):
    ret = 0
    owns = False
    buystate = None
    for state in states:
        change, bollinger = state
        if owns:
            ret = (1 + ret) * (1 + change) - 1
        prestate = (change, bollinger, owns, buystate, ret, None)
        dprestate = discretize(prestate)
        if random.random() < randomness:
            allowed_actions = actions_filter(dprestate)
            action = random.choice(list(allowed_actions))
        else:
            *_, action = learner.argmax(dprestate)
        action = RACTION[action]
        yield change, bollinger, owns, buystate, ret, action
        if action == 'buy':
            owns = True
            buystate = bollinger
        elif action == 'sell':
            owns = False
            buystate = None
            ret = 0

def reward(state):
    ret = 0
    action = state[COL['action']]
    if action == 'sell':
        ret = state[COL['return']]
    return ret

def actions_filter(dstate):
    if dstate[DCOL['owns']] == 1:
        return (ACTION['sell'], ACTION['promise'],)
    else:
        return (ACTION['wait'], ACTION['buy'],)

def discretize(state):
    # change, bollinger, owns, buystate, return, action
    _, bollinger, owns, buystate, _, action = state
    return (
        BOLLINGER[bollinger],
        discretize_owns(owns),
        BOLLINGER[str(buystate)],
        discretize_action(action),
    )


class Learner(trading.rl.Q):

    def __init__(self, table=None):
        shape = (
            len(BOLLINGER.keys()),
            2,
            len(BOLLINGER.keys()),
            len(ACTION.keys()),
        )
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
        owns = False
        buystate = None
        for change, bollinger in states:
            if owns:
                ret = (1 + ret) * (1 + change) - 1
            state = (change, bollinger, owns, buystate, ret, None,)
            *_, action = self.argmax(self.discretize(state))
            if action == ACTION['buy']:
                owns = True
                ret = 0
                buystate = bollinger
            elif action == ACTION['sell']:
                owns = False
                total_return += ret
                ret = 0
                buystate = None
            if output:
                print('output', state, RACTION[action])

        # print('total_return', total_return)
        return total_return, ret
