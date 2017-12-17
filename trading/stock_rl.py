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

# index for owning
OWN = {
    'false': 'none',
    'true': 'own',
}

ROWN = {
    'none': 0,
    'own': 1,
}

# index for action
ACTION = {
    'promise': 'promise',
    'buy': 'buy',
    'sell': 'sell',
    'wait': 'wait',
}

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
    '3': 'wait',
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
    # return None in case invalid action
    return DACTION.get(action)

def discretize_owns(owns):
    if owns == 'none':
        return 0
    else:
        return 1
# states is % change, bollinger, owns, return, action
#                       b
#            s                       -
#            b       -           s       -
#          s      b              b -   s   -
#               s              s          s

def state_generator(states, horizon=10, sample_rate=1):
    leafs_at = trading.lib.tree.Searcher.leafs_at
    Node = trading.lib.tree.Node
    buy_root = tree = Node(ACTION['buy'])
    for i in range(1, horizon):
        for child in leafs_at(tree, level=i):
            if random.random() < sample_rate:
                continue
            if child.value in [ACTION['sell'], ACTION['wait']]:
                child.add_child(Node(ACTION['wait']))
                child.add_child(Node(ACTION['buy']))
            elif child.value in [ACTION['buy'], ACTION['promise']]:
                child.add_child(Node(ACTION['sell']))
                child.add_child(Node(ACTION['promise']))

    wait_root = tree = Node(ACTION['wait'])
    for i in range(1, horizon):
        for child in leafs_at(tree, level=i):
            if random.random() < sample_rate:
                continue
            if child.value in [ACTION['sell'], ACTION['wait']]:
                child.add_child(Node(ACTION['wait']))
                child.add_child(Node(ACTION['buy']))
            elif child.value in [ACTION['buy'], ACTION['promise']]:
                child.add_child(Node(ACTION['promise']))
                child.add_child(Node(ACTION['sell']))

    results = lambda: itertools.chain(
        trading.lib.tree.Searcher.search(buy_root, value=ACTION['sell']),
        trading.lib.tree.Searcher.search(wait_root, value=ACTION['sell'])
    )

    for i in range(len(states) + 1 - horizon):
        state_chunk = states[i:i + horizon]
        for node in results():
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
                elif action == ACTION['sell']:
                    owns = OWN['false']

def reward(state):
    ret = 0
    action = state[COL['action']]
    if action == ACTION['sell']:
        ret = state[COL['return']]
    return ret

def actions_filter(state):
    if state[COL['owns']] == ROWN['own']:
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
        shape = (12, 3, 2, 12, 4,)
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
            b = 'mid'
            if bollinger == 1:
                b = 'abv'
            elif bollinger == 2:
                b = 'bel'
            if action == DACTION['buy']:
                owns = OWN['true']
                ret = 0
            elif action == DACTION['sell']:
                owns = OWN['false']
                total_return += ret
                ret = 0
        print('total return', total_return)
