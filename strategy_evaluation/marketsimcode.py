"""
Student Name: William Callan
GT User ID: wcallan3
GT ID: 903546349
"""

import datetime as dt
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data


def author():
    return "wcallan3"


def compute_portvals(
        trades_df,
        symbol="JPM",
        start_val=100000,
        commission=9.95,
        impact=0.005
):
    """
    Computes the portfolio values.

    :param trades_df: Dataframe of the orders
    :type trades_df: pandas.DataFrame
    :param symbol: The stock being traded
    :type symbol: str
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # sort order rows by date
    trades = trades_df.copy()
    trades = trades.sort_index()
    # get the start and end dates for trading
    sd = trades.index[0]
    ed = trades.index[-1]
    # get the adj. close values for the stocks being traded
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna()
    # create a column filled with 1s, useful later when calculating values
    prices["Cash"] = 1
    # set cash column of trades to -1 * price * trades
    trades["Cash"] = -1 * prices[symbol] * trades[symbol]
    # get the total holdings of the stock and cash
    holdings = trades.cumsum()
    holdings["Cash"] += start_val
    # get values of stocks: holdings * prices
    values = holdings * prices
    # get our portfolio value by summing each row
    portvalues = values.sum(axis=1)

    # apply commission and impact to our cash on hand whenever we make a transaction
    # TODO: Could this be vectorized?
    for date, row in trades[trades[symbol] != 0].iterrows():
        # commission
        portvalues[date:] -= commission
        # impact
        price = prices.loc[date][symbol]
        shares = np.abs(row[symbol])
        portvalues[date:] -= impact * price * shares

    return portvalues


if __name__ == "__main__":
    print("marketsimcode")
