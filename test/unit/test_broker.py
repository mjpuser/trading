import trading.broker
import unittest
import test.unit.data

class BrokerTestCase(unittest.TestCase):

    def setUp(self):
        self.broker = trading.broker.Broker(3)

    def test_add_one_stock(self):
        self.assertEqual(len(self.broker.holdings), 0)
        self.broker.add('AAPL')
        self.assertEqual(len(self.broker.holdings), 1)

    def test_add_too_many_stocks(self):
        with self.assertRaises(Exception) as context:
            for _ in range(4):
                self.broker.add('TEST')
        self.assertEqual(context.exception.message, 'Too many stocks')

    def test_find_sells_no_sales(self):
        # take data, given current holdings, return symbols to sell
        pass
