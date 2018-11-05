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
import csv
import numpy as np

p =argparse.ArgumentParser()
p.add_argument('--output_dir', type=str, default="./")
p.add_argument('--test_data', type=str, default="/home/shapelim/git_files/cee554/lstm_codes/inputs/np_test_data_2.csv")
#In case of test 1
# p.add_argument('--uni', type=str, default= "/home/shapelim/KRoC_Results/1104_uni_poly.csv")
# p.add_argument('--bi', type=str, default= "/home/shapelim/KRoC_Results/1104_bi_poly.csv")
# p.add_argument('--multimodal_uni', type=str, default= "/home/shapelim/KRoC_Results/1104_unimul_poly.csv")
# p.add_argument('--multimodal_bi', type=str, default= "/home/shapelim/KRoC_Results/1104_bimul_poly.csv")

p.add_argument('--save_MSE_name', type=str, default="Distance_error_result__test2.png")
p.add_argument('--save_error_percent_name', type=str, default="test_stack.png")
p.add_argument('--save_trajectory_name', type=str, default="Test_trajectory11.png") #""Trajectory_result_refined_interval_10_smoothed_test_stack.png")
p.add_argument('--data_interval', type=int, default= 21)

args = p.parse_args()
FILE_NAME = '1105_np_test2_result'
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

test_csv = np.loadtxt(args.test_data, delimiter = ',')

##################################################
#                 Initialize
##################################################
uwb_list = []
for uwb_position in UWB_LOC:
    uwb = UWBAnchor(uwb_position)
    uwb_list.append(uwb)
MCL = MonteCarloLocalization()
viz = Visualization(args)
result_container = DataContainer()
MSE = 0
##################################################
#                     PF
##################################################
for i in range(len(test_csv)):#1143): #(2318):
    range_data = test_csv[:,:-3]

    for j, uwb in enumerate(uwb_list):
        uwb.setRange(range_data[i, j])

    z = MCL.calculateDistanceDiff(uwb_list)
    MCL.setWeights(z)
    particles_list, weights_list = MCL.resampling()
    position = MCL.getEstimate(particles_list, weights_list)
    MCL.scatterParticleWithLiDAR(particles_list,0.68)

    result_container.append_data(position.x, position.y, position.z)

    print ("step :  %d"%i)
##################################################
#              Set files names
##################################################
csv_file_name = args.output_dir + FILE_NAME + '.csv'
png_file_name = args.output_dir + FILE_NAME +".png"

##################################################
#         Set csv files & result.png
##################################################
result_file = open(csv_file_name, 'w', encoding='utf-8', newline='')
wr = csv.writer(result_file)
for i in range(len(result_container.x_axis)):
    wr.writerow([result_container.x_axis[i], result_container.y_axis[i], result_container.z_axis[i]])
result_file.close()

##################################################
#          Set png file name & draw
##################################################
viz.set_3D_plot_name(png_file_name)
viz.drawResult3D(csv_file_name)

viz._calDistanceError3D(csv_file_name)


