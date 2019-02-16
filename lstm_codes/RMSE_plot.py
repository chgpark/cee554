from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np
import os
from scipy import interpolate
from scipy.interpolate import spline

'''
b blue
g green
r red
c cyan 
m magenta
y yellow
k balck
w white
'''
# COLORSET = [(0,0,1), 'g', 'r', 'm', 'c', 'y'] #, 'k','w']
COLORSET = [(241/255.0, 101/255.0, 65/255.0), (19/255.0, 163/255.0, 153/255.0), (2/255.0, 23/255.0, 157/255.0), (191/255.0, 17/255.0, 46/255.0)]
SOFT_COLORSET = [(241/255.0, 187/255.0, 165/255.0), (174/255.0, 245/255.0, 231/255.0), (115/255.0, 123/255.0, 173/255.0), (232/255.0, 138/255.0, 139/255.0)]
LINE = ['-', ':', '--', '-']
# LABEL = ['LSTM', 'GRU', 'Bi-LSTM', 'Stacked Bi-LSTM']
LABEL = ['3 anchors', '5 anchors', '8 anchors']
  #marker o x + * s:square d:diamond p:five_pointed star


def plot_RMSE_plot():
    global COLORSET, LINE, LABEL
    # plot_title = "RMSE according to sequence length"
    # plt.title(plot_title)
    # plt.rcParams['Figure.figsize'] = (14, 3)
    # plt.figure(figsize=(7,4.326))
    anchor_3_rmse = [4.963, 4.860, 4.89, 4.729, 4.466]
    anchor_5_rmse = [3.526, 3.468, 3.408, 3.400, 3.210]
    anchor_8_rmse = [3.223, 3.127, 3.212, 3.072, 3.115]
    x_axis = [2, 3, 5, 8, 12]
    i=0
    plt.plot(x_axis, anchor_3_rmse, color= COLORSET[i], marker = 'o', linestyle = LINE[i],label = LABEL[i])
    i=1
    plt.plot(x_axis, anchor_5_rmse, color= COLORSET[i], marker = '*', linestyle = LINE[i],label = LABEL[i])
    i=2
    plt.plot(x_axis, anchor_8_rmse, color= COLORSET[i], marker = 'd', linestyle = LINE[i],label = LABEL[i])
    plt.legend()
    plt.grid(True)
    plt.xlim(1, 13)
    # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
    # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
    plt.ylim(3,5)
    plt.xlabel("Sequence length", fontsize=15)
    plt.ylabel("RMSE [cm]", fontsize=15)
    fig = plt.gcf()
    # plt.show()
    fig.savefig('RMSE_error.png')
    print ("Done")

if __name__ == "__main__":
    plot_RMSE_plot()
