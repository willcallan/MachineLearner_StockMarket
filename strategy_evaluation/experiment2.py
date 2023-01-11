"""
Student Name: William Callan
GT User ID: wcallan3
GT ID: 903546349
"""

import os
import numpy as np
import pandas as pd
import datetime as dt

from util import get_data
import StrategyLearner as sl
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt


def author():
    return "wcallan3"


def run_experiment(
    symbol="JPM",
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 12, 31),
    sv=100000,
    commission=9.95
):
    impact_values = (0.0000, 0.0075, 0.0150)
    # create the three learners with different impact values
    learner_A = sl.StrategyLearner(impact=impact_values[0], commission=commission)
    learner_B = sl.StrategyLearner(impact=impact_values[1], commission=commission)
    learner_C = sl.StrategyLearner(impact=impact_values[2], commission=commission)
    # train the learners on in-sample data
    learner_A.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner_B.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner_C.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    # test the learners on in-sample data
    trades_A = learner_A.testPolicy(symbol=symbol, sd=sd, ed=ed)
    trades_B = learner_B.testPolicy(symbol=symbol, sd=sd, ed=ed)
    trades_C = learner_C.testPolicy(symbol=symbol, sd=sd, ed=ed)

    # calculate portfolio values for each learner
    portval_A = compute_portvals(trades_df=trades_A, symbol=symbol, start_val=sv, commission=commission, impact=impact_values[0])
    portval_B = compute_portvals(trades_df=trades_B, symbol=symbol, start_val=sv, commission=commission, impact=impact_values[1])
    portval_C = compute_portvals(trades_df=trades_C, symbol=symbol, start_val=sv, commission=commission, impact=impact_values[2])
    portval_A = portval_A / portval_A[0]
    portval_B = portval_B / portval_B[0]
    portval_C = portval_C / portval_C[0]
    # calculate some other stats
    cumret_A = (portval_A[-1] / portval_A[0]) - 1
    cumret_B = (portval_B[-1] / portval_B[0]) - 1
    cumret_C = (portval_C[-1] / portval_C[0]) - 1
    dailyret_A = get_daily_returns(portval_A)
    dailyret_B = get_daily_returns(portval_B)
    dailyret_C = get_daily_returns(portval_C)
    stddev_A = dailyret_A.std()
    stddev_B = dailyret_B.std()
    stddev_C = dailyret_C.std()
    mean_A = dailyret_A.mean()
    mean_B = dailyret_B.mean()
    mean_C = dailyret_C.mean()
    numtrades_A = trades_A[trades_A != 0].dropna().shape[0]
    numtrades_B = trades_B[trades_B != 0].dropna().shape[0]
    numtrades_C = trades_C[trades_C != 0].dropna().shape[0]

    # plot the portfolio values
    fig, ax = plt.subplots()
    # graph the benchmark against the manual strategy
    ax.plot(portval_A, label=f"{impact_values[0]}", color="purple")
    ax.plot(portval_B, label=f"{impact_values[1]}", color="red")
    ax.plot(portval_C, label=f"{impact_values[2]}", color="blue")
    # title and labels
    plt.title(f"Strategy Learner with Varying Impact Costs")
    plt.xlabel("Date")
    fig.autofmt_xdate()
    plt.ylabel("Portfolio Value")
    # formatting
    plt.grid()
    plt.legend()
    # save / show
    plt.savefig(f"images/experiment2_cumret.png")
    #plt.show()
    plt.clf()

    # log the results
    output_file = open(f"p8_experiment2_results.txt", "w")
    output_file.write("--Part 3.3.4--")
    output_file.write(f"\nImpact {impact_values[0]}:")
    output_file.write(f"\n  Cum. Ret.: {'{:.6f}'.format(cumret_A)}")
    output_file.write(f"\n  Std. Dev.: {'{:.6f}'.format(stddev_A)}")
    output_file.write(f"\n       Mean: {'{:.6f}'.format(mean_A)}")
    output_file.write(f"\n     Trades: {numtrades_A}")
    output_file.write(f"\nImpact {impact_values[1]}:")
    output_file.write(f"\n  Cum. Ret.: {'{:.6f}'.format(cumret_B)}")
    output_file.write(f"\n  Std. Dev.: {'{:.6f}'.format(stddev_B)}")
    output_file.write(f"\n       Mean: {'{:.6f}'.format(mean_B)}")
    output_file.write(f"\n     Trades: {numtrades_B}")
    output_file.write(f"\nImpact {impact_values[2]}:")
    output_file.write(f"\n  Cum. Ret.: {'{:.6f}'.format(cumret_C)}")
    output_file.write(f"\n  Std. Dev.: {'{:.6f}'.format(stddev_C)}")
    output_file.write(f"\n       Mean: {'{:.6f}'.format(mean_C)}")
    output_file.write(f"\n     Trades: {numtrades_C}")
    output_file.close()


def get_daily_returns(df):
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


if __name__ == "__main__":
    run_experiment()
