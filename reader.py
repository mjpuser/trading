import trading.stock_rl
import pandas as pd

learner = trading.stock_rl.Learner()
df = pd.read_csv('/Users/mattp/Downloads/GOOG.csv')
df['change'] = df['Close'].pct_change()
train_data = [ (x,) for x in df['change'][-90:] ]

learner.learn(
    lambda: trading.stock_rl.state_generator(train_data, horizon=10),
)

test_data = df['change'][-90:]

learner.predict(test_data)
