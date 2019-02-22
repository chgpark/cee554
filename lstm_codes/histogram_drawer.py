import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
import collections
import numpy as np
import os

from scipy.stats import norm
import matplotlib.mlab as mlab


def extract_distances_from_pozyx_csv(pozyx_csv):
    '''
    In pozyx_csv, sync is not distorted,
    so, if you check by yourself,
    then print below commented_line
    '''
    dist_0 = '-'
    dist_1 = 'RSS'
    dist_2 = 'dist.1'
    dist_3 = '-.2'

    # print(pozyx_csv['-'].iloc[i])
    # print(pozyx_csv['RSS'].iloc[i])
    # print(pozyx_csv['dist.1'].iloc[i])
    # print(pozyx_csv['-.2'].iloc[i])
    # print(pozyx_csv['RSS.2'].iloc[i])
    # print(pozyx_csv['dist.3'].iloc[i])
    # print(pozyx_csv['-.4'].iloc[i])
    # print(pozyx_csv['RSS.4'].iloc[i])

    pandas_index_list = [dist_0, dist_1, dist_2, dist_3]
    distance_0 = np.array(pozyx_csv[dist_0])
    distance_1 = np.array(pozyx_csv[dist_1])
    distance_2 = np.array(pozyx_csv[dist_2])
    distance_3 = np.array(pozyx_csv[dist_3])


    return distance_0, distance_1, distance_2, distance_3

if __name__  == "__main__":
    '''Setting directories of object files'''
    # full_path = "/home/shapelim/Desktop/uio_net_data/190108/"
    # dir = full_path + "test/2019-01-08-08-31-38/"
    abs_path = "/home/shapelim/Downloads/0220_calibration/test2_1/_slash_pozyx_slash_range.csv"
    pozyx_csv = pd.read_csv(abs_path, delimiter=',')
    d0, d1, d2, d3 = extract_distances_from_pozyx_csv(pozyx_csv)

    d_list = [d0, d1, d2, d3]

    for sensor_idx, d in enumerate(d_list):
        # Set target d
        n_bins = 18
        x = d

        for i in range(len(x)):
            if x[i] == 0:
                x[i] = (x[i-1] + x[i+1])/2

        # This process is essential to get n for normalizing pdf...
        # I don't know what it is
        n, bins, patches = plt.hist(x, bins=n_bins, color=(0.0, 0.0, 0.5), density=True)

        plt.cla()
        fig = plt.figure()

        weights = np.ones_like(x)/float(len(x))
        plt.hist(x, bins=n_bins, color=(0.0, 0.0, 0.5), weights=weights)

        ''' For gaussian fitting'''
        xt = plt.xticks()[0]
        xmin, xmax = min(xt), max(xt)
        lnspc = np.linspace(xmin, xmax, len(x))

        # lets try the normal distribution first
        m, s = norm.fit(x)  # get mean and standard deviation
        pdf_g = norm.pdf(lnspc, m, s)  # now get theoretical values in our interval
        pdf_g = pdf_g/sum(n)
        plt.plot(lnspc, pdf_g, label="Norm", color='r', linestyle='--')  # plot it
        print(m, s)
        plt.title(r'$ \mu=100,\ \sigma=15$')
        plt.title(r'$ \mu={},\ \sigma={}$'.format(round(m, 3), round(s, 3)))
        # plt.title("Mean: " + str(round(m, 2)) + "Std: " + str(round(s,2)))
        plt.xlabel("Measured distance [m]")
        plt.ylabel("Probability density")

        # (mu, sigma) = norm.fit(x)
        # gaussian_line = mlab.normpdf(n_bins, mu, sigma)
        # plt.plot(n_bins, gaussian_line, 'r', linewidth=2)

        fig = plt.gcf()
        fig.savefig("histo2{}.png".format(sensor_idx))
