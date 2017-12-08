import unittest
import trading.lib.tree


class TreeTestCase(unittest.TestCase):

    def setUp(self):
        self.tree = trading.lib.tree.Tree()

    def test_child_node_is_in_children(self):
        node = trading.lib.tree.Node()
        self.tree.add_child(node)
        self.assertTrue(node in self.tree.children)

    def test_child_node_has_parent(self):
        node = trading.lib.tree.Node()
        self.tree.add_child(node)
        self.assertEquals(node.parent, self.tree)

    def test_node_with_no_children_is_empty_list(self):
        self.assertEquals(self.tree.children, [])

    def test_node_has_a_value(self):
        value = 'test'
        node = trading.lib.tree.Node(value)
        self.assertEquals(node.value, value)

    def test_node_is_leaf(self):
        value = 'test'
        node = trading.lib.tree.Node(value)
        self.assertTrue(node.is_leaf())

    def test_node_is_not_leaf(self):
        value = 'test'
        node = trading.lib.tree.Node(value)
        self.assertTrue(node.is_leaf())
        node.add_child(node)
        self.assertFalse(node.is_leaf())

    def test_only_accept_child_node_type(self):
        value = None
        with self.assertRaises(ValueError) as context:
            self.tree.add_child(value)
        self.assertEquals(
            context.exception.message,
            'Child node must be type trading.lib.tree.Node not {}'.format(value.__class__.__name__)
        )
        self.assertTrue(isinstance(context.exception, ValueError))

    def test_accept_tree_dict_structure(self):
        structure = {
            'value': 1,
            'children': [
                { 'value': 2 },
                { 'value': 3 },
                {
                    'value': 4,
                    'children': [
                        { 'value': 5 }
                    ]
                }
            ]
        }
        tree = trading.lib.tree.Tree(structure)
        self.assertTrue(tree.value, 1)
        self.assertTrue(tree.children[0].value, 2)
        self.assertTrue(tree.children[1].value, 3)
        self.assertTrue(tree.children[2].value, 4)
        self.assertTrue(tree.children[2].children[0].value, 5)


class TreeTraverseTestCase(unittest.TestCase):

    def setUp(self):
        self.searcher = trading.lib.tree.Searcher()
        self.tree = trading.lib.tree.Tree()

    def build_tree(self):
        #               1
        #          /        \
        #      2               6
        #    /   \         /   \    \
        #  3      4      7       9    10
        #        /        \
        #       5          8
        structure = {
            'value': 1,
            'children': [
                { 'value': 2, 'children': [
                    { 'value': 3 },
                    { 'value': 4, 'children': [
                        { 'value': 5 }
                    ] }
                ] },
                { 'value': 6, 'children': [
                    { 'value': 7, 'children': [
                        { 'value': 8 }
                    ] },
                    { 'value': 9 },
                    { 'value': 10 }
                ] }
            ]
        }
        tree = trading.lib.tree.Tree(structure)
        return tree

    def test_depth_first_search(self):
        tree = self.build_tree()
        results = [ x.value for x in self.searcher.search(tree) ]
        self.assertEquals(results, [ x for x in range(1, 11) ])

    def test_option_to_emit_paths(self):
        tree = self.build_tree()
        paths = [
            (1, 2, 3,),
            (1, 2, 4, 5,),
            (1, 6, 7, 8,),
            (1, 6, 9,),
            (1, 6, 10,),
        ]
        results = [ tuple(map(lambda x: x.value, x)) for x in self.searcher.paths(tree) ]
        self.assertEquals(results, paths)
