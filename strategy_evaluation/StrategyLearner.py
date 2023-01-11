""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: William Callan  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT User ID: wcallan3  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT ID: 903546349  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

import datetime as dt
import pandas as pd
import numpy as np

from util import get_data
import QLearner as ql
import indicators as ind


class StrategyLearner(object):
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        If verbose = False your code should not generate ANY output.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type verbose: bool  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type impact: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type commission: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        # parameters
        self.bins = 5
        self.starting_rar = 0.5
        self.iterations = 3
        # initialize the Q-Learner
        num_states = 3 * (self.bins ** 3)
        self.qlearner = ql.QLearner(
            num_states=num_states,
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=self.starting_rar,
            radr=0.99,
            dyna=300,
            verbose=verbose
        )
        # States: A state has 4 digits: H, S, P, M, concatenated to the number HSPM
        #   Total num_states: H*S*P*M
        #   H [0, 2]: Our holdings: 0 for 0, 1 for -1000, 2 for 1000
        #   S [0, bins-1]: SMA value
        #   P [0, bins-1]: PPO value
        #   M [0, bins-1]: Momentum value
        # Actions: 0 is do nothing, 1 is short until holdings=-1000, 2 is long until holdings=1000

    def author(self):
        return "wcallan3"

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    ):
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Trains your strategy learner over a given time frame.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param symbol: The stock symbol to train on  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type symbol: str  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sd: datetime.datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type ed: datetime.datetime
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sv: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        # initialize the Q-Learner
        num_states = 3 * (self.bins ** 3)
        self.qlearner = ql.QLearner(
            num_states=num_states,
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=self.starting_rar,
            radr=0.99,
            dyna=200,
            verbose=self.verbose
        )

        # get stock prices
        prices = get_data(symbols=[symbol], dates=pd.date_range(sd, ed), addSPY=False)
        prices = prices.dropna()

        # get the indicators
        sma = ind.sma(symbol=symbol, sd=sd, ed=ed)
        ppo = ind.ppo(symbol=symbol, sd=sd, ed=ed)
        momentum = ind.momentum(symbol=symbol, sd=sd, ed=ed)
        sma.fillna(0, inplace=True)
        ppo.fillna(0, inplace=True)
        momentum.fillna(0, inplace=True)
        # discretize the indicators
        sma_discretized = self.get_discretized_df(sma)
        ppo_discretized = self.get_discretized_df(ppo)
        momentum_discretized = self.get_discretized_df(momentum)

        # iteratively train over the data a few times
        for repeat in range(self.iterations):
            # set the initial state for the Q-learner and get its first action
            state = self.get_state_integer(0, sma_discretized.iat[0], ppo_discretized.iat[0], momentum_discretized.iat[0])
            action = self.qlearner.querysetstate(state)
            # create the empty trades df
            trades = prices.copy()
            trades[symbol] = 0
            # update the trades with the action taken by the learner
            self.update_trades_df(trades=trades, action=action, date=trades.index[0])

            losses = 0
            profit = 0
            old_value = sv
            # add evidence to the learner
            for i in range(1, trades.shape[0]):
                date = trades.index[i]
                traded_shares = trades.iat[i-1, 0]
                shares_value = traded_shares * prices.iat[i-1, 0]
                holdings = int(trades.sum())
                # compute the portfolio value after the action to get our reward
                profit -= shares_value
                if traded_shares != 0:
                    losses -= self.commission
                    losses -= shares_value * self.impact
                assets = holdings * prices.iat[i, 0]
                new_value = sv + assets + profit + losses
                # reward should be daily return
                reward = (new_value / old_value) - 1
                old_value = new_value
                # compute the new state we are in
                state = self.get_state_integer(
                    holding=holdings,
                    sma=sma_discretized.at[date],
                    ppo=ppo_discretized.at[date],
                    momentum=momentum_discretized.at[date]
                )
                # query our Q-learner
                action = self.qlearner.query(s_prime=state, r=reward)
                # update the trades with the action taken by the learner
                self.update_trades_df(trades=trades, action=action, date=date)


    def get_discretized_thresholds(self, df):
        """
        Returns the bin threshold values for a given dataframe
        """
        binsize = df.shape[0] / self.bins
        df_sorted = df.sort_values(by=[df.columns[0]])
        df_thresholds = np.zeros(self.bins + 1)

        for i in range(0, self.bins - 1):
            df_thresholds[i+1] = df_sorted.ix[int((i+1) * binsize)]
        df_thresholds[0] = np.NINF
        df_thresholds[self.bins] = np.inf

        return df_thresholds

    def get_discretized_df(self, df):
        thr = self.get_discretized_thresholds(df)
        discretized = pd.cut(df[df.columns[0]], thr, labels=np.arange(self.bins))

        return discretized

    def get_state_integer(self, holding, sma, ppo, momentum):
        # convert holdings to int
        holding_int = 0
        if holding == -1000:
            holding_int = 1
        elif holding == 1000:
            holding_int = 2

        return momentum + self.bins * ppo + (self.bins ** 2) * sma + (self.bins ** 3) * holding_int

    def update_trades_df(self, trades, action, date):
        # get the target number of shares to hold
        if action == 0:
            return
        elif action == 1:
            target_holdings = -1000
        else:
            target_holdings = 1000
        # get the currently held number of shares
        current_holdings = int(trades.sum())
        # find the difference
        trades.at[date] = target_holdings - current_holdings

    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):
        """
        Tests your learner using data outside of the training data  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type symbol: str  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sd: datetime.datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type ed: datetime.datetime
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sv: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :rtype: pandas.DataFrame  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        # get stock prices
        prices = get_data(symbols=[symbol], dates=pd.date_range(sd, ed), addSPY=False)
        prices = prices.dropna()

        # get the indicators
        sma = ind.sma(symbol=symbol, sd=sd, ed=ed)
        ppo = ind.ppo(symbol=symbol, sd=sd, ed=ed)
        momentum = ind.momentum(symbol=symbol, sd=sd, ed=ed)
        sma.fillna(0, inplace=True)
        ppo.fillna(0, inplace=True)
        momentum.fillna(0, inplace=True)
        # discretize the indicators
        sma_discretized = self.get_discretized_df(sma)
        ppo_discretized = self.get_discretized_df(ppo)
        momentum_discretized = self.get_discretized_df(momentum)

        # create the empty trades df
        trades = prices.copy()
        trades[symbol] = 0

        old_rar = self.qlearner.rar
        self.qlearner.rar = 0

        # query the learner at each state
        for date in trades.index:
            # compute the new state we are in
            holdings = int(trades.sum())
            state = self.get_state_integer(
                holding=holdings,
                sma=sma_discretized.at[date],
                ppo=ppo_discretized.at[date],
                momentum=momentum_discretized.at[date]
            )
            # query our Q-learner
            action = self.qlearner.querysetstate(state)
            # update the trades with the action taken by the learner
            self.update_trades_df(trades=trades, action=action, date=date)

        self.qlearner.rar = old_rar

        return trades


if __name__ == "__main__":
    sl = StrategyLearner()
    sl.add_evidence()
    t = sl.testPolicy()
    print(t)
