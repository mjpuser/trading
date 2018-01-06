#!/usr/bin/env python

import ijson
import json
import numpy as np
import sys

items = ijson.items(sys.stdin, 'results.item')
print('{ "results": [')
first = True
for item in items:
    if not first:
        print(',')
    else:
        first = False
    for attr in ['open', 'high', 'low', 'close']:
        item[attr] = float(item[attr])
    avg = np.average([item['high'], item['low'], item['close']])
    item['moneyFlow'] = int(avg * item['volume'])
    print(json.dumps(item, indent=2))
print(']}')
