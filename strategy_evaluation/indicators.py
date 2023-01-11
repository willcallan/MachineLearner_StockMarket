"""
Student Name: William Callan
GT User ID: wcallan3
GT ID: 903546349
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from util import get_data, plot_data


def author():
    return "wcallan3"


def bollinger_bands(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        window=12
):
    # get the prices on trade days
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna()

    # get the rolling mean (SMA) and rolling std
    sma = prices.rolling(window).mean()
    r_std = prices.rolling(window).std()
    # get the upper and lower bollinger bands
    bb_upper = sma + 2 * r_std
    bb_lower = sma - 2 * r_std
    # standardize the return vector
    #   when price is at top bb, vector has value 1
    #   when price is at bottom bb, vector has value -1
    price_standardized = (prices - sma) / (2 * r_std)

    return price_standardized


def sma(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        window=20
):
    # get the prices on trade days
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna()

    # get the SMA and its diff to price
    sma = prices.rolling(window).mean()
    diff = prices - sma

    # return normalized vector
    return diff / diff.std()


def momentum(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        shift=20
):
    # get the prices on trade days
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna()

    # get the SMA for small and large windows
    momentum = prices / prices.shift(shift) - 1

    # return normalized vector
    return momentum / momentum.std()


def ppo(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        windows=(12, 26, 9)
):
    # parameters
    window_short = windows[0]
    window_long = windows[1]
    window_ppo = windows[2]

    # get the prices on trade days
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna()
    # get the EMA for small and large windows
    ema_short = prices.ewm(span=window_short, adjust=True).mean()
    ema_long = prices.ewm(span=window_long, adjust=True).mean()
    # calculate the PPO, signal line, and their difference
    ppo = 100 * (ema_short - ema_long) / ema_long
    signal = ppo.ewm(span=window_ppo, adjust=True).mean()
    diff = ppo - signal

    # return normalized vector
    return diff / diff.std()


def stochastic(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        windows=(14, 3)
):
    # parameters
    window_size = windows[0]
    window_stoch = windows[1]

    # get the prices on trade days
    # NOTE: Use close, NOT adjusted close
    prices_close = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname="Close")
    prices_high = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname="High")
    prices_low = get_data([symbol], pd.date_range(sd, ed), addSPY=False, colname="Low")
    prices_close = prices_close.dropna()
    prices_high = prices_high.dropna()
    prices_low = prices_low.dropna()
    # get the highest/lowest prices over window period
    rolling_high = prices_high.rolling(window_size).max()
    rolling_low = prices_low.rolling(window_size).min()
    # calculate the Stochastic Oscillator and its difference
    stoch = 100 * (prices_close - rolling_low) / (rolling_high - rolling_low)
    rolling_stoch = stoch.rolling(window_stoch).mean()
    diff = stoch - rolling_stoch

    # return normalized vector
    return diff / diff.std()


def plot_together(symbol, sd, ed, bb, mo, pp, sm, st):
    # get the prices on trade days
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices = prices.dropna()

    # plot the values
    fig, ax = plt.subplots()
    #ax[0].plot(prices, label="Price", color="C1")
    ax.plot(sm / sm.std(), label="SMA")
    ax.plot(pp / pp.std(), label="PPO")
    ax.plot(mo / mo.std(), label="Momentum")
    # title and labels
    #fig.suptitle("Indicators", fontweight="bold")
    #ax[0].set_title("Price")
    ax.set_title("Comparisons")
    plt.xlabel("Date")
    fig.autofmt_xdate()
    #ax[0].set_ylabel("Price")
    ax.set_ylabel("Value")
    #plt.ylim([-1, 1])
    # formatting
    #ax[0].grid()
    ax.grid()
    fig.legend()
    # save / show
    #plt.savefig("images/indicators_ppo.png")
    plt.show()
    plt.clf()


def generate_all_plots(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31)
):
    bb = bollinger_bands(symbol=symbol, sd=sd, ed=ed)
    mo = momentum(symbol=symbol, sd=sd, ed=ed)
    pp = ppo(symbol=symbol, sd=sd, ed=ed)
    sm = sma(symbol=symbol, sd=sd, ed=ed)
    st = stochastic(symbol=symbol, sd=sd, ed=ed)

    plot_together(symbol, sd, ed, bb, mo, pp, sm, st)


if __name__ == "__main__":
    generate_all_plots(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31)
    )
