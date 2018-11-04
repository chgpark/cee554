#!/usr/bin/env python
import random
import pandas as pd
from math import *
###############
from mcl import MonteCarloLocalization
from mcl import UWBAnchor
from plot_result import Visualization
from environment_settings import UWB_LOC, MAP_X, MAP_Y, MAP_Z
from environment_settings import SAMPLING_NUM, UNCERTAINTY, SENSOR_NOISE
from environment_settings import DIFFERENCE_TAG_LiDAR
from environment_settings import MEASUREMENT_LiDAR_WEIGHT
import argparse
import numpy as np

p =argparse.ArgumentParser()
p.add_argument('--output_dir', type=str, default="./")
p.add_argument('--test_data', type=str, default="inputs/poly_3D.csv")
# p.add_argument('--gt_dir', type=str, default="inputs/test_data_diagonal_curve2D.csv")
#In case of test 1
p.add_argument('--uni', type=str, default= "/home/shapelim/KRoC_Results/1104_uni_poly.csv")
p.add_argument('--bi', type=str, default= "/home/shapelim/KRoC_Results/1104_bi_poly.csv")
p.add_argument('--multimodal_uni', type=str, default= "/home/shapelim/KRoC_Results/1104_unimul_poly.csv")
p.add_argument('--multimodal_bi', type=str, default= "/home/shapelim/KRoC_Results/1104_bimul_poly.csv")

p.add_argument('--save_MSE_name', type=str, default="Distance_error_result__test2.png")
p.add_argument('--save_error_percent_name', type=str, default="test_stack.png")
p.add_argument('--save_trajectory_name', type=str, default="Test_trajectory11.png") #""Trajectory_result_refined_interval_10_smoothed_test_stack.png")
p.add_argument('--data_interval', type=int, default= 21)

args = p.parse_args()
INPUTDATA = "/home/shapelim/git_files/cee554/lstm_codes/inputs/np_test_data_1.csv"

OUTPUTDIR = "./" + str(SAMPLING_NUM) + "_" + str(UNCERTAINTY).replace('.','_')

class DataContainer:
    def __init__(self):
        self.x_axis = []
        self.y_axis = []
        self.z_axis = []

    def append_data(self,x,y,z):
        self.x_axis.append(x)
        self.y_axis.append(y)
        self.z_axis.append(z)

test_pd = np.loadtxt(INPUTDATA, delimiter = ',')

uwb_list = []
for uwb_position in UWB_LOC:
    uwb = UWBAnchor(uwb_position)
    uwb_list.append(uwb)

MCL = MonteCarloLocalization()
viz = Visualization(args)

result_container = DataContainer
# range_data = test_pd.loc[1,:'d16']
# print (range_data)
MSE = 0
for i in range(len(test_pd)-1):#1143): #(2318):

    range_data = INPUTDATA[:,:-3]

    # print (range_data)
    for j, uwb in enumerate(uwb_list):
        uwb.setRange(range_data[j])

    z = MCL.calculateDistanceDiff(uwb_list)
    MCL.setWeights(z)
    particles_list, weights_list = MCL.resampling()
    position = MCL.getEstimate(particles_list, weights_list)
    result_container.append_data(position.x, position.y, position.z)

    print ("step :  %d"%i)

# MSE /= 400
# print (MSE)




