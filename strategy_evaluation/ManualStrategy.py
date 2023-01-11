"""
Student Name: William Callan
GT User ID: wcallan3
GT ID: 903546349
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from util import get_data
from marketsimcode import compute_portvals
import indicators as ind


class ManualStrategy(object):
    # constructor
    def __init__(self, verbose=False, commission=9.95, impact=0.005):
        """
        Constructor method
        """
        self.verbose = verbose
        self.commission = commission
        self.impact = impact

    def author(self):
        return "wcallan3"

    def get_stock_prices(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
    ):
        # get the prices on trade days
        prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
        prices = prices.dropna()

        return prices

    def get_daily_returns(self, df):
        """
        Computes the daily returns for a stock dataframe.

        :param df: Stock price values
        :type df: pandas.DataFrame
        :return: Stock daily returns
        :rtype: pandas.DataFrame
        """
        daily_returns = (df / df.shift(1)) - 1
        daily_returns.ix[0] = 0
        daily_returns.index.name = "Date"
        return daily_returns

    def get_benchmark_trades(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31)
    ):
        # get prices for stock
        prices = self.get_stock_prices(symbol=symbol, sd=sd, ed=ed)

        # create benchmark trades with buy 1000 on first day
        trades = prices.copy()
        trades[symbol] = 0
        trades[symbol][0] = 1000

        return trades

    def testPolicy(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31)
    ):
        # get prices for stock
        prices = self.get_stock_prices(symbol=symbol, sd=sd, ed=ed)
        # get the indicators to use (replace nan with 0)
        sma = ind.sma(symbol=symbol, sd=sd, ed=ed, window=20)
        ppo = ind.ppo(symbol=symbol, sd=sd, ed=ed, windows=(12, 26, 9))
        momentum = ind.momentum(symbol=symbol, sd=sd, ed=ed, shift=20)
        sma.fillna(0, inplace=True)
        ppo.fillna(0, inplace=True)
        momentum.fillna(0, inplace=True)
        # convert the indicators to signed representations
        #   when an indicator crosses the 0 line:
        #   signed has a value of +2 OR -2
        signed_sma = np.sign(sma).astype({symbol: 'int'})
        signed_ppo = np.sign(ppo).astype({symbol: 'int'})
        signed_momentum = np.sign(momentum).astype({symbol: 'int'})
        signed_sma = -1 * (signed_sma - signed_sma.shift(1))
        signed_ppo = -1 * (signed_ppo - signed_ppo.shift(1))
        signed_momentum = -1 * (signed_momentum - signed_momentum.shift(1))
        signed_sma.fillna(0, inplace=True)
        signed_ppo.fillna(0, inplace=True)
        signed_momentum.fillna(0, inplace=True)
        signed_sum = signed_sma + signed_ppo + signed_momentum

        # create the empty trades dataframe
        trades = prices.copy()
        trades[symbol] = 0
        # fill in the trades based on the indicators (only when they are != 0)
        current_holdings = 0
        for date, row in signed_sum[signed_sum[symbol] != 0].iterrows():
            # set the target holdings for the stock based on indicators
            fut_ret = row[symbol]
            if fut_ret > 0:
                target_holdings = 1000
            else:
                target_holdings = -1000
            # buy / sell stock until we hit target holdings
            trades.ix[date] = target_holdings - current_holdings
            current_holdings = target_holdings

        return trades

    def plot_trades(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000
    ):
        # get the portfolio values for benchmark and ms, normalized
        b_trades = self.get_benchmark_trades(symbol=symbol, sd=sd, ed=ed)
        b_vals = compute_portvals(trades_df=b_trades, symbol=symbol, start_val=sv, commission=self.commission, impact=self.impact)
        b_vals_normed = b_vals / b_vals[0]
        ms_trades = self.testPolicy(symbol=symbol, sd=sd, ed=ed)
        ms_vals = compute_portvals(trades_df=ms_trades, symbol=symbol, start_val=sv, commission=self.commission, impact=self.impact)
        ms_vals_normed = ms_vals / ms_vals[0]

        # calculate additional stats for benchmark and ms
        b_returns = self.get_daily_returns(b_vals)
        b_cumret = b_vals[-1] / b_vals[0] - 1
        b_stdev = b_returns.std()
        b_mean = b_returns.mean()
        ms_returns = self.get_daily_returns(ms_vals)
        ms_cumret = ms_vals[-1] / ms_vals[0] - 1
        ms_stdev = ms_returns.std()
        ms_mean = ms_returns.mean()
        # log the results
        output_file = open(f"p8_manualstrategy_results_{sd.year}-{ed.year}.txt", "w")
        output_file.write("--Part 3.3.1--")
        output_file.write("\nBenchmark:")
        output_file.write(f"\n  Cum. Ret.: {'{:.6f}'.format(b_cumret)}")
        output_file.write(f"\n  Std. Dev.: {'{:.6f}'.format(b_stdev)}")
        output_file.write(f"\n       Mean: {'{:.6f}'.format(b_mean)}")
        output_file.write("\nMS:")
        output_file.write(f"\n  Cum. Ret.: {'{:.6f}'.format(ms_cumret)}")
        output_file.write(f"\n  Std. Dev.: {'{:.6f}'.format(ms_stdev)}")
        output_file.write(f"\n       Mean: {'{:.6f}'.format(ms_mean)}")
        output_file.close()

        # plot the portfolio values
        fig, ax = plt.subplots()
        # mark where we went long/short with vertical lines
        longs = ms_trades[ms_trades[symbol] > 0]
        shorts = ms_trades[ms_trades[symbol] < 0]
        for date, row in longs.iterrows():
            plt.axvline(date, color="blue", alpha=0.5)
        for date, row in shorts.iterrows():
            plt.axvline(date, color="black", alpha=0.5)
        # graph the benchmark against the manual strategy
        ax.plot(b_vals_normed, label="Benchmark", color="purple")
        ax.plot(ms_vals_normed, label="Manual Strategy", color="red")
        # title and labels
        plt.title(f"Trading Strategy Portfolio Values ({sd.year}-{ed.year})")
        plt.xlabel("Date")
        fig.autofmt_xdate()
        plt.ylabel("Portfolio Value")
        # formatting
        plt.grid()
        plt.legend()
        # save / show
        plt.savefig(f"images/manualstrategy_{sd.year}-{ed.year}.png")
        plt.clf()

        return ms_trades

    def generate_plots(self):
        self.plot_trades(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31))
        self.plot_trades(sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31))


if __name__ == "__main__":
    ms = ManualStrategy()
    ms.generate_plots()
