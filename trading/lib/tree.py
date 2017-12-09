
class Node(object):

    def __init__(self, value=None, children=None):
        self.children = children if type(children) == list else []
        self.value = value
        self.parent = None

    def add_child(self, node):
        if isinstance(node, Node):
            self.children.append(node)
            node.parent = self
        else:
            node_type = '{}.{}'.format(__name__, Node.__name__)
            curr_type = getattr(node, '__module__', '')
            class_name = node.__class__.__name__
            if curr_type is not '':
                curr_type += '.'
            curr_type += class_name
            raise ValueError('Child node must be type {} not {}'.format(node_type, curr_type))

    def is_leaf(self):
        return len(self.children) == 0

    def path(self):
        path = []
        node = self
        while node is not None:
            path.insert(0, node)
            node = node.parent
        return tuple(path)


class Tree(Node):

    def __init__(self, structure=None):
        super(Tree, self).__init__()
        if structure is not None:
            self.value = structure.get('value')
            for child in structure.get('children', []):
                self.add_child(Tree(child))


class Searcher:

    def search(self, node):
        yield node
        for child in node.children:
            for child in self.search(child):
                yield child

    def leafs_at(self, node, level=1):
        if level == 1 and node.is_leaf():
            yield node
        elif level > 1:
            for child in node.children:
                for leaf in self.leafs_at(child, level - 1):
                    yield leaf
