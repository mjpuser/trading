import trading.stock_rl
import pandas as pd
import os

window_size = 20
num_of_std = 1.2
test_size = 365
iterations = 5000
randomness = 1
alpha = 0.001
gamma = 1
#filename = 'goog.npy'
filename = None
filepath = './qtables/{}'.format(filename)
stock = 'VZ'

learner = trading.stock_rl.Learner()
#df = pd.read_csv('/Users/mattp/Downloads/AMD.csv')
#df = pd.read_csv('/Users/mattp/Downloads/GOOG.csv')
#df = pd.read_csv('/Users/mattp/Downloads/SPY.csv')
#df = pd.read_csv('/Users/mattp/Downloads/BTC-USD.csv')
#df = pd.read_csv('/Users/mattp/Downloads/TSLA.csv')
df = pd.read_csv('/Users/mattp/Downloads/{}.csv'.format(stock))

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

# changes = []
# def listify(row):
#     changes.append(row['change'])
#     return row['change']
#
# df[:].apply(listify, axis=1)
# s = sorted(changes)
# print(s)
# for i in range(126, 1260, 126):
#     print(i, s[i])
learner.predict(test_data)
count = 0
xdata = []
ydata = []
def callback():
    global count
    ret = learner.predict(test_data)
    xdata.append(count)
    ydata.append(ret)
    # learner.store('./qtables/{}-nnnewtable'.format(count))
    count = count + 1
    if count < 100:
        print('iteration', count, 'return', ret, 'rando', randomness)
    if count % (iterations / 100.0) == 0:
        print('iteration', count, 'return', ret, 'rando', randomness)
    #if count == iterations:
        # learner.store('./qtables/goog')


def state_gen():
    global randomness
    randomness = randomness / 1.0001
    return trading.stock_rl.state_generator(train_data, learner, randomness)

learner.learn(
    state_gen,
    iterations=iterations,
    callback=callback,
    alpha=alpha,
    gamma=gamma
)


from bokeh.plotting import figure, output_file, show

# output to static HTML file
output_file("graphs/lines.html")

# create a new plot with a title and axis labels
p = figure(title=stock, x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
p.line(xdata, ydata, legend="Temp.", line_width=2)

# show the results
show(p)
