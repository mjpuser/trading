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
    'bollinger': 0,
    'owns': 1,
    'buystate': 2,
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
    'dipped': 2, # went from above to inside
    'above': 3, # above the band
    'returned': 4, # went from below to inside
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
    if dstate[DCOL['owns']] == 1 and dstate[DCOL['bollinger']] == BOLLINGER['dipped']:
        return (ACTION['sell'],)
    elif dstate[DCOL['owns']] == 1:
        return (ACTION['promise'],)
    elif dstate[DCOL['owns']] == 0 and dstate[DCOL['bollinger']] == BOLLINGER['returned']:
        return (ACTION['buy'],)
    else:
        return (ACTION['wait'],)


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
        buys = []
        sells = []
        for i, (change, bollinger,) in enumerate(states):
            if owns:
                ret = (1 + ret) * (1 + change) - 1
            state = (change, bollinger, owns, buystate, ret, None,)
            *_, action = self.argmax(self.discretize(state))
            if action == ACTION['buy']:
                owns = True
                ret = 0
                buystate = bollinger
                buys.append(i)
            elif action == ACTION['sell']:
                owns = False
                total_return += ret
                ret = 0
                buystate = None
                sells.append(i)
            if output:
                print('output', state, RACTION[action])

        # print('total_return', total_return)
        return total_return, ret, buys, sells

    def is_buy(self, bollinger, momentum):
        return bollinger == 'returned'

    def is_sell(self, bollinger, momentum):
        return bollinger == 'dipped'

def calculate_return(state):
    change, _, _, _, ret, *_ = state
    return (1 + ret) * (1 + change) - 1
