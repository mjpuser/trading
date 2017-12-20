import trading.stock_rl
import pandas as pd
import os

window_size = 20
num_of_std = 1.5
test_size = 365
horizon = 40
iterations = 4
sample_rate = 0.6
filename = 'goog-45.npy'
#filename = None
filepath = './qtables/{}'.format(filename)

learner = trading.stock_rl.Learner()
#df = pd.read_csv('/Users/mattp/Downloads/GOOG.csv')
df = pd.read_csv('/Users/mattp/Downloads/SPY.csv')
#df = pd.read_csv('/Users/mattp/Downloads/BTC-USD.csv')
#df = pd.read_csv('/Users/mattp/Downloads/TSLA.csv')

if os.path.isfile(filepath):
    learner.load(filepath)

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
test_data = df[-test_size:].apply(get_data, axis=1)
learner.predict(test_data)

# learner.learn(
#     lambda: trading.stock_rl.state_generator(train_data, horizon, sample_rate),
#     iterations=iterations,
#     callback=callback
# )
#learner.predict(test_data)
#learner.store('./qtables/newtable')
