
import trading.lib.tree

def state_generator(horizon=10):
    tree = trading.lib.tree.Tree({
        'value': 'buy',
        'children': [
            { 'value': 'sell' }
        ]
    })
    root = tree
    for _ in range(horizon - 2):
        node = trading.lib.tree.Tree({
            'value': None,
            'children': [
                { 'value': 'sell' }
            ]
        })
        tree.add_child(node)
        tree = node

    for node in trading.lib.tree.Searcher.search(root, value='sell'):
        yield tuple(map(lambda x: x.value, node.path()))
