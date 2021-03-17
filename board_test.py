#!/usr/bin/env python3

from board import HarvestEnv
import matplotlib.pyplot as plt


if __name__=="__main__":
    harvest = HarvestEnv(10, (10, 10))

    fig, ax = plt.subplots()

    print(harvest.render(ax))
