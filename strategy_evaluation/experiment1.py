"""
Student Name: William Callan
GT User ID: wcallan3
GT ID: 903546349
"""

import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt

from util import get_data
from marketsimcode import compute_portvals
import ManualStrategy as ms
import StrategyLearner as sl


def author():
    return "wcallan3"


def get_trader_objects(
    symbol="JPM",
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 12, 31),
    sv=100000,
    commission=9.95,
    impact=0.05
):
    # get the manual and learner objects
    ms_obj = ms.ManualStrategy(commission=commission, impact=impact)
    sl_obj = sl.StrategyLearner(commission=commission, impact=impact)
    # train the learner
    sl_obj.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)

    return ms_obj, sl_obj


def plot_trades(
    ms_obj,
    sl_obj,
    symbol="JPM",
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 12, 31),
    sv=100000,
    commission=9.95,
    impact=0.05
):
    # get the benchmark, manual, and learner trades for the time period
    benchmark_trades = ms_obj.get_benchmark_trades(symbol=symbol, sd=sd, ed=ed)
    ms_trades = ms_obj.testPolicy(symbol=symbol, sd=sd, ed=ed)
    sl_trades = sl_obj.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    # retrieve and normalize the portfolio values for each strategy
    benchmark_vals = compute_portvals(trades_df=benchmark_trades, symbol=symbol, start_val=sv, commission=commission, impact=impact)
    ms_vals = compute_portvals(trades_df=ms_trades, symbol=symbol, start_val=sv, commission=commission, impact=impact)
    sl_vals = compute_portvals(trades_df=sl_trades, symbol=symbol, start_val=sv, commission=commission, impact=impact)

    benchmark_vals = benchmark_vals / benchmark_vals[0]
    ms_vals = ms_vals / ms_vals[0]
    sl_vals = sl_vals / sl_vals[0]

    # plot the portfolio values
    fig, ax = plt.subplots()
    # graph the benchmark against the manual strategy
    ax.plot(benchmark_vals, label="Benchmark", color="purple")
    ax.plot(ms_vals, label="Manual Strategy", color="red")
    ax.plot(sl_vals, label="Strategy Learner", color="blue")
    # title and labels
    plt.title(f"Trading Strategy Portfolio Values ({sd.year}-{ed.year})")
    plt.xlabel("Date")
    fig.autofmt_xdate()
    plt.ylabel("Portfolio Value")
    # formatting
    plt.grid()
    plt.legend()
    # save / show
    plt.savefig(f"images/experiment1_{sd.year}-{ed.year}.png")
    #plt.show()
    plt.clf()


def run_experiment():
    ms_obj, sl_obj = get_trader_objects(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
        commission=9.95,
        impact=0.005
    )
    #for i in range(1):
    #    sl_obj.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    plot_trades(
        ms_obj,
        sl_obj,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
        commission=9.95,
        impact=0.005
    )
    plot_trades(
        ms_obj,
        sl_obj,
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000,
        commission=9.95,
        impact=0.005
    )


if __name__ == "__main__":
    run_experiment()
