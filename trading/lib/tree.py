
class Node:

    def __init__(self, value=None, children=None):
        self.children = children if type(children) == list else []
        self.value = value

    def add_child(self, node):
        if isinstance(node, Node):
            self.children.append(node)
        else:
            curr_type = getattr(node, '__module__', '')
            class_name = node.__class__.__name__
            if curr_type is not '':
                curr_type += '.'
            curr_type += class_name
            raise ValueError('Child node must be type {} not {}'.format(Node, curr_type))

class Tree(Node):
    pass
