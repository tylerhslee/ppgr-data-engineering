import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("data/std/ml_input.csv", header=0, index_col=None)
ppgr = dataset["ppgr"]

std_ppgr = (ppgr - ppgr.mean()) / ppgr.std()

# ax2 = std_ppgr.plot.hist(bins=3)
# plt.show()

spread = std_ppgr.max() - std_ppgr.min()
intervals = [
    (std_ppgr.min(), std_ppgr.min() + spread / 3),
    (std_ppgr.min() + spread / 3, std_ppgr.min() + spread * 2 / 3),
    (std_ppgr.min() + spread * 2 / 3, std_ppgr.max() + 1e-8)
]

def rank_ppgr(ppgr):
    for i in range(len(intervals)):
        if (ppgr >= intervals[i][0]) and (ppgr < intervals[i][1]):
            return i

dataset["ppgr_rank"] = std_ppgr.apply(rank_ppgr)

print(dataset.head())

dataset.to_csv("data/std/ml_input_rank.csv")
