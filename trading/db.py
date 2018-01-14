import datetime
import sqlite3

conn = sqlite3.connect('data/trading.db')
conn.row_factory = sqlite3.Row # return dicts for records

COLUMNS = """
date,
close,
macd,
macds,
boll,
boll_ub,
boll_lb
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
            {}_stock
        WHERE
            date = ?
            AND close is not null
            AND boll_ub is not null
            AND boll_lb is not null
            AND boll is not null 
    """.format(COLUMNS, symbol)
    return c.execute(sql, [day]).fetchone()

def get_last_stock(symbol, day, limit):
    sql = """
        SELECT
            {}
        FROM
            {}_stock
        WHERE
            date <= ?
            AND close is not null
            AND boll_ub is not null
            AND boll_lb is not null
            AND boll is not null 
        ORDER BY
            date DESC
        LIMIT {}
    """.format(COLUMNS, symbol, limit)
    return c.execute(sql, [day]).fetchmany(int(limit))

def get_stocks():
    sql = """
        SELECT
            replace(name, '_stock', '') as name
        FROM
            sqlite_master
        WHERE
            type = 'table'
    """
    for record in c.execute(sql).fetchall():
        yield record['name']
