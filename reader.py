import trading.stock_rl
import pandas as pd

window_size = 20
num_of_std = 1.5
test_size = 100
horizon = 16
iterations = 1

learner = trading.stock_rl.Learner()
#df = pd.read_csv('/Users/mattp/Downloads/GOOG.csv')
df = pd.read_csv('/home/matthew/Downloads/SPY.csv')
#df = pd.read_csv('/Users/mattp/Downloads/BTC-USD.csv')
#df = pd.read_csv('/Users/mattp/Downloads/TSLA.csv')

stock_price = df['Close']
df['change'] = df['Adj Close'].pct_change()

rolling_mean = stock_price.rolling(window=window_size).mean()
rolling_std = stock_price.rolling(window=window_size).std()
df['upper'] = upper_band = rolling_mean + (rolling_std * num_of_std)
df['lower'] = lower_band = rolling_mean - (rolling_std * num_of_std)

def bollinger(row):
    x = 0
    if row['upper'] < row['Close']:
        x = 1
    elif row['lower'] > row['Close']:
        x = 2
    return x

df['bollinger'] = df.apply(bollinger, axis=1)

get_data = lambda r: (r['change'], r['bollinger'])
train_data = df[window_size:-test_size].apply(get_data, axis=1)
learner.learn(
    lambda: trading.stock_rl.state_generator(train_data, horizon),
    iterations=iterations
)

test_data = df[-test_size:].apply(get_data, axis=1)

learner.predict(test_data)
