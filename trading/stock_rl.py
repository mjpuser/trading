import trading.lib.tree
import itertools
import trading.rl

# index to dimension
COL = {
    'change': 0,
    'bollinger': 1,
    'owns': 2,
    'return': 3,
    'action': 4,
}

# index for owning
OWN = {
    'false': 0,
    'true': 1,
}

# index for action
ACTION = {
    'noop': 0,
    'buy': 1,
    'sell': 2,
}

def discretize_return(ret):
    return discretize_change(ret)

def discretize_change(change):
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
    return ACTION.get(action)

# states is % change, bollinger, return, owns, action
def state_generator(states, horizon=10):
    buy_sell = tree = trading.lib.tree.Tree({
        'value': ACTION['buy'],
        'children': [
            { 'value': ACTION['sell'] }
        ]
    })
    buy_root = tree
    for _ in range(horizon - 2):
        node = trading.lib.tree.Tree({
            'value': ACTION['noop'],
            'children': [
                { 'value': ACTION['sell'] }
            ]
        })
        tree.add_child(node)
        tree = node

    noop_root = tree = trading.lib.tree.Tree({
        'value': ACTION['noop']
    })
    for _ in range(horizon - 2):
        buy_sell = trading.lib.tree.Tree({
            'value': ACTION['buy'],
            'children': [
                { 'value': ACTION['sell'] }
            ]
        })
        noop = trading.lib.tree.Tree({
            'value': ACTION['noop']
        })
        tree.add_child(buy_sell)
        tree.add_child(noop)
        tree = noop

    results = itertools.chain(
        trading.lib.tree.Searcher.search(buy_root, value=ACTION['sell']),
        trading.lib.tree.Searcher.search(noop_root, value=ACTION['sell'])
    )

    for i in range(len(states) - horizon + 1):
        state_chunk = states[i:i + horizon]
        for node in results:
            actions = list(map(lambda n: (n.value,), node.path()))
            chunk = state_chunk[0:len(actions)]
            episode = map(
                lambda s: itertools.chain.from_iterable(s),
                zip(chunk, actions)
            )

            owns = OWN['false']
            ret = 0
            for state in episode:
                change, bollinger, action = state
                if owns == OWN['true']:
                    ret = (1 + ret) * (1 + change) - 1

                yield change, bollinger, owns, ret, action
                if action == ACTION['buy']:
                    owns = OWN['true']
                    ret = 0

def reward(state):
    action = state[COL['action']]
    if action == ACTION['sell']:
        return state[COL['return']]
    return 0

def actions_filter(state):
    if state[COL['owns']] == OWN['true']:
        return (ACTION['noop'], ACTION['sell'],)
    else:
        return (ACTION['noop'], ACTION['buy'],)

def discretize(state):
    change, bollinger, owns, ret, action = state
    return (
        discretize_change(change),
        bollinger,
        owns,
        discretize_return(ret),
        action,
    )


class Learner(trading.rl.Q):

    def __init__(self, table=None):
        shape = (12, 3, 2, 12, 3,)
        super(Learner, self).__init__(
            reward,
            shape,
            discretize,
            actions_filter,
            table
        )

    def predict(self, states):
        total_return = 0
        ret = 0
        owns = OWN['false']
        for change, bollinger in states:
            if owns == OWN['true']:
                ret = (1 + ret) * (1 + change) - 1
            *_, action = self.argmax(self.discretize((change, bollinger, owns, ret, None,)))
            print(ret, bollinger, action)
            if action == ACTION['buy']:
                owns = OWN['true']
                ret = 0
            elif action == ACTION['sell']:
                owns = OWN['false']
                total_return += ret
                ret = 0
        print('total return', total_return)
