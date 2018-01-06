import trading.stock_rl
import pandas as pd
import os
import sys

window_size = 20
num_of_std = 1.5
test_size = 252 # number of trading days in a year
iterations = 1
randomness = 0
delta_randomness = 1.005
alpha = 0.0001
gamma = 1
#filename = 'goog.npy'
filename = None
filepath = './qtables/{}'.format(filename)
stock = sys.argv[1] or 'FULT'

learner = trading.stock_rl.Learner()
df = pd.read_csv('data/history/{}.csv'.format(stock))

if os.path.isfile(filepath):
    learner.load(filepath)

stock_price = df['close']
df['change'] = df['close'].pct_change()

rolling_mean = stock_price.rolling(window=window_size).mean()
rolling_std = stock_price.rolling(window=window_size).std()
df['upper'] = upper_band = rolling_mean + (rolling_std * num_of_std)
df['lower'] = lower_band = rolling_mean - (rolling_std * num_of_std)

previous_bollinger = None
def bollinger(row):
    global previous_bollinger
    x = 'inside'
    if row['upper'] < row['close']:
        x = 'above'
    elif row['lower'] > row['close']:
        x = 'below'

    if previous_bollinger == 'above' and x == 'inside':
        x = 'dipped'
    elif previous_bollinger == 'below' and x == 'inside':
        x = 'returned'

    previous_bollinger = x

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
total, ret, buys, sells = learner.predict(test_data)
print(stock, 'total: {}, return: {}'.format(total, ret))

# count = 0
# xdata = []
# ydata = []
# def callback():
#     global count
#     output = (count + 1 == iterations)
#     total, ret = learner.predict(test_data, output)
#     xdata.append(count)
#     ydata.append(total + ret)
#     # learner.store('./qtables/{}-nnnewtable'.format(count))
#     count = count + 1
#     if count < 100:
#         print('iteration', count, 'total', total, 'holding', ret, 'rando', randomness)
#     if count % (iterations / 100.0) == 0:
#         print('iteration', count, 'total', total, 'holding', ret, 'rando', randomness)
#     #if count == iterations:
#         # learner.store('./qtables/goog')
#
#
# def state_gen():
#     global randomness
#     randomness = randomness / delta_randomness
#     return trading.stock_rl.state_generator(train_data, learner, randomness)
#
# learner.learn(
#     state_gen,
#     iterations=iterations,
#     callback=callback,
#     alpha=alpha,
#     gamma=gamma
# )

from bokeh.plotting import figure, output_file, show
from bokeh.models import Span

# output to static HTML file
output_file("graphs/{}.html".format(stock))

# create a new plot with a title and axis labels
p = figure(title=stock, x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
graph_data = df[-test_size:]
ydata = list(graph_data['close'])
xdata = [x for x in range(test_size)]
p.line(xdata, ydata, legend="Temp.", line_width=2)

for buy in buys:
    buy_span = Span(location=buy, dimension='height', line_color='green', line_dash='dashed', line_width=1)
    p.add_layout(buy_span)

for sell in sells:
    sell_span = Span(location=sell, dimension='height', line_color='red', line_dash='dashed', line_width=1)
    p.add_layout(sell_span)
# show the results
show(p)


# map raw data to data with technical indicators
# read processed data for indicators
# randomly choose top 20 (depending on how many currently holding)
# if choose, this enables the holding map to calculate return
# the map to calculate return dumps to a central file return.csv
# there will be a column per holding
# to find out the total return, sum the last row
