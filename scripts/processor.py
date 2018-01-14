#!/usr/bin/env python

import pandas as pd
import stockstats
import sqlite3
import sys

symbols = sys.stdin.readlines()
conn = sqlite3.connect('data/trading.db')

for symbol in symbols:
    try:
        symbol = symbol.strip()
        print('Importing {}'.format(symbol))
        df = pd.read_csv('data/history/{}.csv'.format(symbol))
        stockstats.StockDataFrame.BOLL_STD_TIMES = 1
        stock = stockstats.StockDataFrame.retype(df)
        stock = stock[['close','macd','macds','adx','boll','boll_ub','boll_lb']]
        stock.to_sql(name='{}_stock'.format(symbol), con=conn)
    except Exception as e:
        print('Failed to import {}'.format(symbol))
        print(e)

