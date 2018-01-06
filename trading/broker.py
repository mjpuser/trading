import trading.stock_rl

class Broker:

    def __init__(self, size=20):
        self.size = size
        self.holdings = {}
        self.learner = trading.stock_rl.Learner()

    def add(self, stock):
        # adds stock to one of the holders
        pass

    def choose(self, stocks):
        # chooses a single stock out of the potentials randomly
        pass

    def expand_state(self, stock, state):
        if stock in self.holdings.keys():
            exapnded = self.holdings[stock]
        else:
            change, bollinger = state
            ret = 0
            owns = False
            buystate = None
            action = None
            state = (change, bollinger, owns, buystate, ret, action)
        return expanded

    def find_sells(self, holdings_states):
        sells = []
        for symbol, state in holdings_states:
            if self.learner.is_sell(state):
                sells.append(symbol)
        return sells

    def crunch(self, stock_data):
        # stock data is all of today's stock data
        # find out if we have to sell any of our current stocks
        # map stock data for the stocks in holding
        # sell them if we do
        # if you have room for to buy more stocks, look for some buy indicators
        # randomly choose some if we have room
        # all holdings print out return percentage
        pass


# raw data downloaded
# find potential sells
# make sells
# map raw holdings -> processed holdings
# map raw solds -> processed solds
# choose buys (can't be in holdings or just sold)
# map raw buys -> processed buys/holdings
# map remaining raw -> processed (can't process because you don't know whats going to be sold)
# output holdings and returns
