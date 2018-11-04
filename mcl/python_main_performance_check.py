#!/usr/bin/env python
import random
import pandas as pd
from math import *
###############
from mcl import MonteCarloLocalization
from mcl import UWBAnchor
from drawer import Visualization
from drawer import DataContainer

from environment_settings import UWB_LOC, MAP_X, MAP_Y, MAP_Z
from environment_settings import SAMPLING_NUM, UNCERTAINTY, SENSOR_NOISE
from environment_settings import DIFFERENCE_TAG_LiDAR
from environment_settings import MEASUREMENT_LiDAR_WEIGHT


# INPUTDATA = "/home/shapelim/git_files/ComplexDisaster/180907_pohang/2018-09-07-pohang-uwb/uwb_only/uwb_rectified.csv"
# ZDATA = '/home/shapelim/git_files/ComplexDisaster/180907_pohang/2018-09-07-pohang-uwb/uwb_only/slash_fcc_info.csv'

# INPUTDATA = "/home/shapelim/git_files/ComplexDisaster/180903_12anchors/2018-09-03-23-57-39/uwb_rectified.csv"
# ZDATA = '/home/shapelim/git_files/ComplexDisaster/180903_12anchors/2018-09-03-23-57-39/_slash_fcc_info.csv'

INPUTDATA = "/home/shapelim/git_files/ComplexDisaster/20180808/non_fog_range_data_80/uwb_rectified.csv"
ZDATA = '/home/shapelim/git_files/ComplexDisaster/20180808/non_fog_range_data_80/_slash_fcc_info.csv'

OUTPUTDIR = "/home/shapelim/git_files/mcl_results/0917_with_Gaussian_test1" + str(SAMPLING_NUM) + "_" + str(UNCERTAINTY).replace('.','_')



test_pd = pd.read_csv(INPUTDATA, delimiter = ',')
test_pd2 = pd.read_csv(ZDATA, delimiter = ',')

uwb_list = []
for uwb_position in UWB_LOC:
    uwb = UWBAnchor(uwb_position)
    uwb_list.append(uwb)

MCL = MonteCarloLocalization()
MCL2 = MonteCarloLocalization()
MCL3 = MonteCarloLocalization()
viz = Visualization(uwb_list, OUTPUTDIR)

result_container = DataContainer()
result_container2 = DataContainer()
result_container3 = DataContainer()

# range_data = test_pd.loc[1,:'d16']
# print (range_data)
time_stamp = 3
MSE = 0
for i in range(len(test_pd)-1):#1143): #(2318):
    # viz.show(MCL)
    # print ("Hi!")
    '''For 180903'''
    # h_index = int(round(i * (float(len(test_pd2))/len(test_pd))))
    # height_data = test_pd2.loc[h_index, 'data1']
    # h = (-1) * float(height_data.split(',')[6])
    '''For 180808'''
    h_index = int(round(i * (float(len(test_pd2))/len(test_pd))))
    height_data = test_pd2.loc[h_index, 'data'].split(',')
    height_data = height_data[-1][:-3]

    h = (-1)*float(height_data)

    LiDAR_data = 0.09502746*(h**3) - 0.24772325*(h**2) + 1.179433*h - 0.10962938 + DIFFERENCE_TAG_LiDAR

    range_data = test_pd.loc[i,:'d4']

    '''For Pohang test : 16 range is required'''
    # range_data = test_pd.loc[i,:'d16']

    # print (range_data)
    for j, uwb in enumerate(uwb_list):
        uwb.setRange(range_data[j])

    z = MCL.calculateDistanceDiffWithLiDAR(uwb_list, LiDAR_data, MEASUREMENT_LiDAR_WEIGHT)
    MCL.setWeights(z)
    particles_list, weights_list = MCL.resampling()
    position = MCL.getEstimate(particles_list, weights_list)
    if i%time_stamp == 0:
        result_container.append_data(position.x, position.y, position.z)
    # viz.show(MCL)

    MCL.scatterParticleWithLiDAR(particles_list, LiDAR_data)

    # '''MCL2'''
    # z = MCL2.calculateDistanceDiffWithLiDAR(uwb_list, LiDAR_data, 10)
    # z = MCL.calculateDistanceDiffConsideringMultiPath(uwb_list)
    # z = MCL.calculateCosineSimilarity(uwb_list)
    MCL2.setWeightsbyGaussian(uwb_list, LiDAR_data, 1)
    particles_list, weights_list = MCL2.resampling()
    position = MCL2.getEstimate(particles_list, weights_list)

    if i%time_stamp == 0:
        result_container2.append_data(position.x, position.y, position.z)
    # # viz.show(MCL2)
    MCL2.scatterParticleWithLiDAR(particles_list, LiDAR_data)
    #
    # '''MCL3'''
    MCL3.setWeightsbyGaussian(uwb_list, LiDAR_data, 5)
    particles_list, weights_list = MCL3.resampling()
    position = MCL3.getEstimate(particles_list, weights_list)

    if i%time_stamp == 0:
        result_container3.append_data(position.x, position.y, position.z)
    # # z = MCL.calculateDistanceDiff(uwb_list)
    # z = MCL3.calculateDistanceDiffWithLiDAR(uwb_list, LiDAR_data, 10)
    # # z = MCL.calculateDistanceDiffConsideringMultiPath(uwb_list)
    # # z = MCL.calculateCosineSimilarity(uwb_list)
    #
    # MCL3.setWeights(z)
    # particles_list, weights_list = MCL3.resampling()
    # position = MCL3.getEstimate(particles_list, weights_list)
    # result_container3.append_data(position.x, position.y, position.z)
    # # viz.show(MCL3)
    #
    # MCL.scatterParticle(particles_list)
    MCL3.scatterParticleWithLiDAR(particles_list, LiDAR_data)

    # '''MCL4'''
    # z = MCL4.calculateDistanceDiff(uwb_list)
    # # z = MCL.calculateDistanceDiffConsideringMultiPath(uwb_list)
    # # z = MCL.calculateCosineSimilarity(uwb_list)
    #
    # MCL4.setWeights(z)
    # particles_list, weights_list = MCL4.resampling()
    # position = MCL4.getEstimate(particles_list, weights_list)
    # result_container4.append_data(position.x, position.y, position.z)
    #
    # MCL4.scatterParticle(particles_list)
    # # MCL3.scatterParticleWithLiDAR(particles_list, LiDAR_data)


    print ("step :  %d"%i)
# viz.drawResult(result_container.x_axis, result_container.y_axis, result_container.z_axis)
viz.drawResult_on_timestep_for_comparing(result_container, result_container2, result_container3)

# MSE /= 400
# print (MSE)




