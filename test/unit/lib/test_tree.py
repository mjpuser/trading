import unittest
import trading.lib.tree

Searcher = trading.lib.tree.Searcher


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
        self.assertEqual(node.parent, self.tree)

    def test_node_with_no_children_is_empty_list(self):
        self.assertEqual(self.tree.children, [])

    def test_node_has_a_value(self):
        value = 'test'
        node = trading.lib.tree.Node(value)
        self.assertEqual(node.value, value)

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

    def test_node_path(self):
        value = 'test'
        node = trading.lib.tree.Node(value)
        self.tree.add_child(node)
        self.assertEqual((self.tree, node,), node.path())

    def test_only_accept_child_node_type(self):
        value = None
        with self.assertRaises(ValueError) as context:
            self.tree.add_child(value)
        self.assertEqual(
            list(context.exception.args)[0],
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
        self.tree = self.build_tree()

    def build_tree(self):
        #               1
        #          /   /     \
        #      2      6       7
        #    /   \         /   \    \
        #  3      4      8      10   11
        #        /        \
        #       5          9
        structure = {
            'value': 1,
            'children': [
                { 'value': 2, 'children': [
                    { 'value': 3 },
                    { 'value': 4, 'children': [
                        { 'value': 5 }
                    ] }
                ] },
                { 'value': 6 },
                { 'value': 7, 'children': [
                    { 'value': 8, 'children': [
                        { 'value': 9 }
                    ] },
                    { 'value': 10 },
                    { 'value': 11 }
                ] }
            ]
        }
        tree = trading.lib.tree.Tree(structure)
        return tree

    def test_finds_values(self):
        results = [ x.value for x in Searcher.search(self.tree, 5) ]
        self.assertEqual(results, [ 5 ])

    def test_cant_find_values_with_max_limit(self):
        results = [ x.value for x in Searcher.search(self.tree, 5, 3) ]
        self.assertEqual(results, [])

    def test_finds_values_with_max_limit(self):
        results = [ x.value for x in Searcher.search(self.tree, 5, 4) ]
        self.assertEqual(results, [5])

    def test_leafs_at(self):
        paths = [
            3,
            10,
            11,
        ]
        results = [ node.value for node in Searcher.leafs_at(self.tree, 3) ]
        self.assertEqual(results, paths)

    def test_leafs_at_with_paths(self):
        tree = self.tree
        paths = [
            (1, 2, 3,),
            (1, 7, 10,),
            (1, 7, 11,),
        ]
        results = [
            tuple(map(lambda x: x.value, node.path()))
            for node in Searcher.leafs_at(self.tree, 3)
        ]
        self.assertEqual(results, paths)
