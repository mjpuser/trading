import trading.lib.tree

COL = {
    'change': 0,
    'owns': 1,
    'return': 2,
    'action': 3,
}

OWN = {
    'false': 0,
    'true': 1,
}

ACTION = {
    'noop': 0,
    'buy': 1,
    'sell': 2,
}

def state_generator(horizon=10):
    tree = trading.lib.tree.Tree({
        'value': ACTION['buy'],
        'children': [
            { 'value': ACTION['sell'] }
        ]
    })
    root = tree
    for _ in range(horizon - 2):
        node = trading.lib.tree.Tree({
            'value': ACTION['noop'],
            'children': [
                { 'value': ACTION['sell'] }
            ]
        })
        tree.add_child(node)
        tree = node

    for node in trading.lib.tree.Searcher.search(root, value=ACTION['sell']):
        yield tuple(map(lambda x: x.value, node.path()))

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


class Learner(trading.rl.Q):

    def __init__(self, table=None):
        shape = (2, 2, 2, 3,)
        super(Learner, self).__init__(
            reward,
            shape,
            actions_filter,
            table
        )
