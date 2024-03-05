import os
import argparse
import dill as pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from math import pi
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path

matplotlib.use("Agg")

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = 'dotted'
plt.rcParams['font.size'] = 14


def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])


def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1)
                     * (x2 - x1) + x1)
    return sdata


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


class ComplexRadar():
    """
    From: https://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart
    """
    def __init__(self, fig, variables, ranges,
                 metric, devices_test, n_ordinate_levels=4, label_color="black",
                 label_fontsize=9):
        angles = np.arange(0, 360, 360./len(variables)) + 360./len(variables)/2 # added offset to rotate whole thing

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,
                                         labels=variables)

        labels = []
        for label, angle in zip(axes[0].get_xticklabels(), angles):
            x,y = label.get_position()
            lab = axes[0].text(x,y, label.get_text(), transform=label.get_transform(),
                        ha=label.get_ha(), va=label.get_va(),
                        color=label_color, fontsize=label_fontsize)

            lab.set_rotation(angle - 90 if angle < 180 else angle + 90)
            lab.set_fontweight('bold')
            if lab.get_text() in devices_test:
                lab.set_color('red')
            labels.append(lab)
        axes[0].set_xticklabels([])

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            if metric != "hypervolume":
                grid = grid[::-1] # hack to invert grid
                                  # gridlabels aren't reversed
                ax.set_ylim(max(ranges[i]), min(ranges[i]))
            else:
                ax.set_ylim(*ranges[i])
            gridlabel = ["{}".format(round(x,2))
                         for x in grid]
            gridlabel[0] = "" # clean up origin
            gridlabel[-1] = ""

            if i == 0:
                ax.set_rgrids(grid, labels=gridlabel,
                              angle=angles[i])
            else:
                ax.set_rgrids(grid, labels=["" for k in range(len(gridlabel))],
                              angle=angles[i])
            #ax.spines["polar"].set_visible(False)

        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.metric = metric

    def plot(self, data, annotate=False, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        sdata = np.r_[sdata, sdata[0]]
        self.ax.plot(self.angle, sdata, *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

