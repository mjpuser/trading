import datetime
import sqlite3

conn = sqlite3.connect('data/trading.db')
conn.row_factory = sqlite3.Row # return dicts for records

COLUMNS = """
tradingDay,
open,
high,
low,
close,
volume
"""

c = None
def init():
    global c
    c = conn.cursor()

def get_stock(symbol, day=None):
    # can return none.  it will run everyday, and weekends/holidays no tradin
    day = day or datetime.datetime.now().strftime('%Y-%m-%d')
    sql = """
        SELECT
            {}
        FROM
            {}
        WHERE
            tradingDay = ?
    """.format(COLUMNS, symbol)
    return c.execute(sql, [day]).fetchone()

def get_last_stock(symbol, day, limit):
    sql = """
        SELECT
            {}
        FROM
            {}
        WHERE
            tradingDay <= ?
        ORDER BY
            tradingDay DESC
        LIMIT {}
    """.format(COLUMNS, symbol, limit)
    return c.execute(sql, [day]).fetchmany(int(limit))

def get_stocks():
    sql = """
        SELECT
            name
        FROM
            sqlite_master
        WHERE
            type = 'table'
    """
    for record in c.execute(sql).fetchall():
        yield record['name']
