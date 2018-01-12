import datetime
import numpy as np
import trading.stock_rl
import trading.db

trading.db.init()


window = float(20)
std_multiplier = 1.55

def reduce(fn, arr, prev):
    for val in arr:
        prev = fn(prev, val)
    return prev

def ema(prev_ema, curr_price, period_size):
    k = 2.0 / (period_size + 1)
    return (curr_price * k) + (prev_ema * (1 - k))

def macd(prices):
    period_12 = prices[-12:]
    period_26 = prices
    ema_12 = reduce(lambda x, y: ema(x, y, 12), period_12[1:], period_12[0])
    ema_26 = reduce(lambda x, y: ema(x, y, 26), period_26[1:], period_26[0])
    return ema_12 - ema_26

def process_macd(symbol, day):
    long_range = 26
    short_range = 12
    signal_range = 9
    total_rows = long_range + signal_range + 1
    records = trading.db.get_last_stock(symbol, day, total_rows)
    if len(records) == total_rows:
        records.reverse()
        prices = [float(record['close']) for record in records]
        macds = []
        for i in range(signal_range + 1):
            p = prices[i:long_range + i]
            macds.append(macd(p))
        curr_signal = reduce(lambda x, y: ema(x, y, signal_range), macds[2:], macds[1])
        prev_signal = reduce(lambda x, y: ema(x, y, signal_range), macds[1:], macds[0])
        *_, prev_macd, curr_macd = macds
        if curr_signal < curr_macd and prev_signal > prev_macd:
            return 'bull'
        elif curr_signal > curr_macd and prev_signal < prev_macd:
            return 'bear'
        elif curr_macd > prev_macd:
            return 'up'
        elif curr_macd < prev_macd:
            return 'down'
        else:
            return 'unknown'
    else:
        return 0


def bollinger(prices):
    close = prices[-1]
    mean = np.mean(prices)
    std = np.std(prices)
    distance = std * std_multiplier
    upper = mean + distance
    lower = mean - distance

    if close > upper:
        return 'above'
    elif close < lower:
        return 'below'
    return 'inside'

def process_bollinger(symbol, day):
    records = trading.db.get_last_stock(symbol, day, window + 1)
    prices = [float(record['close']) for record in records]
    prices.reverse()
    if len(records) == int(window) + 1:
        prev = bollinger(prices[:-1])
        curr = bollinger(prices[1:])
        if prev == 'above' and curr == 'inside':
            return 'dipped'
        elif prev == 'below' and curr == 'inside':
            return 'returned'
    else:
        return 'inside'

def get_pct_change(symbol, day):
    curr, prev = trading.db.get_last_stock(symbol, day, 2)
    pct_change = (float(curr['close']) - float(prev['close'])) / float(prev['close'])
    return pct_change

def process(symbol, day):
    bollinger = process_bollinger(symbol, day)
    #momentum = process_macd(symbol, day)
    price = float(trading.db.get_stock(symbol, day)['close'])
    return (bollinger, price)

def get_state(symbol, day):
    pct_change = get_pct_change(symbol, day)
    return (pct_change,)

def calculate_return(ret, change):
    return (1 + ret) * (1 + change) - 1

def daterange(start_date, end_date):
    if start_date <= end_date:
        for n in range( ( end_date - start_date ).days + 1 ):
            yield start_date + datetime.timedelta( n )
    else:
        for n in range( ( start_date - end_date ).days + 1 ):
            yield start_date - datetime.timedelta( n )


class Broker:

    def __init__(self, size=20):
        self.size = size
        self.holdings = {}
        self.learner = trading.stock_rl.Learner()
        self.returns = []

    def add(self, symbol):
        # adds stock to one of the holders
        self.holdings[symbol] = 0

    def remove(self, symbol):
        del self.holdings[symbol]

    def choose(self, stocks):
        # chooses a single stock out of the potentials randomly
        pass

    def update_returns(self, day):
        returns = []
        for symbol, ret in self.holdings.items():
            pct_change, = get_state(symbol, day)
            self.holdings[symbol] = pct_change
            returns.append((symbol, ret,))
        return returns

    def sell_stocks(self, day):
        sells = []
        for symbol in self.holdings.keys():
            record = trading.db.get_stock(symbol, day)
            if record is not None:
                indicators = process(symbol, day)
                if self.learner.is_sell(*indicators):
                    print('sell', symbol, indicators)
                    sells.append(symbol)
        for symbol in sells:
            self.remove(symbol)
        return sells

    def buy_stocks(self, day):
        buys = []
        i_map = {}
        if len(self.holdings.keys()) < self.size:
            for symbol in trading.db.get_stocks():
                if symbol not in self.holdings.keys():
                    record = trading.db.get_stock(symbol, day)
                    if record is not None:
                        indicators = process(symbol, day)
                        if self.learner.is_buy(*indicators):
                            i_map[symbol] = indicators
                            buys.append(symbol)
        remaining = self.size - len(self.holdings.keys())
        if len(buys) > 0:
            buys = np.random.choice(buys, np.min([remaining, len(buys)]))
        for symbol in buys:
            print('buy', symbol, i_map[symbol])
            self.add(symbol)
        return buys

    def crunch(self, day=None):
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        day = day or today
        # find out if we have to sell any of our current stocks
        returns = self.update_returns(day)
        sells = self.sell_stocks(day)
        # if you have room for to buy more stocks, look for some buy indicators
        buys = self.buy_stocks(day)
        # all holdings print out return percentage
        return returns

    def iterate(self, start):
        today = datetime.datetime.now()
        start = datetime.datetime.strptime(start, '%Y-%m-%d')
        total_ret = 0
        for day in daterange(start, today):
            day = day.strftime('%Y-%m-%d')
            if trading.db.get_stock('crl', day) is not None:
                print('---------', day, '------------')
                todays_ret = sum([ ret for _, ret in self.holdings.items() ]) / self.size
                total_ret = calculate_return(total_ret, todays_ret)
                returns = self.crunch(day)
                print(self.holdings)
                print('---> return:', total_ret)


# raw data downloaded
# find potential sells
# make sells
# map raw holdings -> processed holdings
# map raw solds -> processed solds
# choose buys (can't be in holdings or just sold)
# map raw buys -> processed buys/holdings
# map remaining raw -> processed (can't process because you don't know whats going to be sold)
# output holdings and returns
