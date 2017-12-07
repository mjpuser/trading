import unittest
import trading.lib.tree


class TreeTestCase(unittest.TestCase):

    def setUp(self):
        self.tree = trading.lib.tree.Tree()

    def test_child_node_is_in_children(self):
        node = trading.lib.tree.Node()
        self.tree.add_child(node)
        self.assertTrue(node in self.tree.children)

    def test_node_with_no_children_is_empty_list(self):
        self.assertEquals(self.tree.children, [])

    def test_node_has_a_value(self):
        value = 'test'
        node = trading.lib.tree.Node(value)
        self.assertEquals(node.value, value)

    def test_only_accept_child_node_type(self):
        value = None
        with self.assertRaises(ValueError) as context:
            self.tree.add_child(value)
        self.assertEquals(
            context.exception.message,
            'Child node must be type trading.lib.tree.Node not {}'.format(value.__class__.__name__)
        )
        self.assertTrue(isinstance(context.exception, ValueError))
