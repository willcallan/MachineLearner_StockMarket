"""
Student Name: William Callan
GT User ID: wcallan3
GT ID: 903546349
"""

import random
import numpy as np

import ManualStrategy as ms
import experiment1 as ex1
import experiment2 as ex2


def author():
    return "wcallan3"


if __name__ == "__main__":
    # Set seed
    gt_id = 903546349
    random.seed(gt_id)
    np.random.seed(gt_id)
    # 3.3.1
    man_strat = ms.ManualStrategy()
    man_strat.generate_plots()
    # 3.3.3
    ex1.run_experiment()
    # 3.3.4
    ex2.run_experiment()
