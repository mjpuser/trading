import trading.stock_rl
import pandas as pd

learner = trading.stock_rl.Learner()
df = pd.read_csv('/Users/mattp/Downloads/GOOG.csv')
df['change'] = df['Close'].pct_change()
train_data = [ (x,) for x in df['change'][-90:] ]

learner.learn(
    lambda: trading.stock_rl.state_generator(train_data, horizon=10),
)
# rolling_mean = stock_price.rolling(window=window_size).mean()
#     rolling_std  = stock_price.rolling(window=window_size).std()
#     upper_band = rolling_mean + (rolling_std*num_of_std)
#     lower_band = rolling_mean - (rolling_std*num_of_std)
#
#     return rolling_mean, upper_band, lower_band

test_data = df['change'][-90:]

learner.predict(test_data)
