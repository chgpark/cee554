from mpl_toolkits.mplot3d import Axes3D
from math import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import random
import os
from environment_settings import MAP_X, MAP_Y, MAP_Z
COLORSET = [(2/255.0, 23/255.0, 157/255.0), (241/255.0, 101/255.0, 65/255.0),
            (19/255.0, 163/255.0, 153/255.0), (191/255.0, 17/255.0, 46/255.0)]
INTERVAL = 5
# LABEL = ['5', '10', '20', 'W/O LiDAR']
# LABEL = ['10', 'W/O LiDAR']
# LABEL = ['z_scatter_w_LiDAR_', 'scatter_w_LiDAR','z_w_LiDAR']
LABEL = ['Estimates', 'Gaussian_weight_1', 'Gaussian_weight_2']
class DataContainer(object):
    def __init__(self):
        self.x_axis = []
        self.y_axis = []
        self.z_axis = []

    def append_data(self,x,y,z):
        self.x_axis.append(x)
        self.y_axis.append(y)
        self.z_axis.append(z)



class MCLVisualization(object):
    def __init__(self, uwb_list, folder_dir):
        self.folder_name = folder_dir
        if not os.path.isdir(self.folder_name):
            os.mkdir(self.folder_name)
        self.count = 0
        self.uwb_x = []
        self.uwb_y = []
        self.uwb_z = []
        for uwb in uwb_list:
            # print (uwb.x, uwb.y, uwb.z)
            self.uwb_x.append(uwb.x)
            self.uwb_y.append(uwb.y)
            self.uwb_z.append(uwb.z)

    def show(self, MCL):
        if not os.path.isdir(self.folder_name):
            os.mkdir(self.folder_name)
        plt.close('all')
        self.count += 1
        if True: #(self.count % INTERVAL == 0):
            self.fig = plt.figure()
            self.ax1 = self.fig.gca(projection = '3d') #add_subplot(111, projection='3d')
            self.X = []
            self.Y = []
            self.Z = []
            for Particle in MCL.particles:
                self.X.append(Particle.x)
                self.Y.append(Particle.y)
                self.Z.append(Particle.z)
            self.ax1.scatter(self.X, self.Y, self.Z)
            self.ax1.scatter(self.uwb_x, self.uwb_y, self.uwb_z, c = 'r', marker = "^", s = 200)

            # plt.xlabel("X_axis")
            # plt.ylabel("Y_axis")
            self.ax1.set_xlim(-2.5, 2.0)
            self.ax1.set_ylim(-2.5, 2.0)
            self.ax1.set_zlim(0, MAP_Z)
            self.ax1.set_xlabel('$X$')# fontsize=20, rotation=150)
            self.ax1.set_ylabel('$Y$')
            self.ax1.set_zlabel('$Z$')#, fontsize=30, rotation=60)
            self.fig = plt.gcf()
            # plt.show()
            # print ("Save" + self.folder_name)
            self.fig.savefig(self.folder_name + "/viz%d.png"%self.count)

    def drawResult(self, X_list, Y_list, Z_list):

        self.fig = plt.figure()
        # plt.subplot(221)
        # self.ax1 = self.fig.gca(projection = '3d') #add_subplot(111, projection='3d')
        # self.ax1.scatter(X_list, Y_list, Z_list)
        plt.subplot(222)
        plt.plot(X_list, Y_list, color = 'b')
        plt.xlabel("X_axis")
        plt.ylabel("Y_axis")
        plt.subplot(223)
        plt.plot(Y_list, Z_list, color ='b')
        plt.xlabel("Y_axis")
        plt.ylabel("Z_axis")
        plt.subplot(224)
        plt.plot(X_list, Z_list, color ='b')
        plt.xlabel("X_axis")
        plt.ylabel("Z_axis")
        self.fig = plt.gcf()
        self.fig.savefig(self.folder_name +"/Results.png")

    def drawResult_on_timestep(self, t_step, X_list, Y_list, Z_list):

        self.fig = plt.figure()
        # plt.subplot(221)
        # self.ax1 = self.fig.gca(projection = '3d') #add_subplot(111, projection='3d')
        # self.ax1.scatter(X_list, Y_list, Z_list)
        plt.subplot(222)
        plt.plot(t_step, X_list, color = 'b')
        plt.xlabel("time step")
        plt.ylabel("X axis")
        plt.subplot(223)
        plt.plot(t_step, Y_list, color ='b')
        plt.xlabel("time step")
        plt.ylabel("Y axis")
        plt.subplot(224)
        plt.plot(t_step, Z_list, color ='b')
        plt.xlabel("time step")
        plt.ylabel("Z_axis")
        self.fig = plt.gcf()
        self.fig.savefig(self.folder_name +"/Results_on_timestep_w_fixed_z_0_6.png")

    def drawResult_on_timestep_for_comparing(self,  *position_results):
        self.fig = plt.figure()
        # plt.subplot(221)
        # self.ax1 = self.fig.gca(projection = '3d') #add_subplot(111, projection='3d')
        # self.ax1.scatter(X_list, Y_list, Z_list)
        plt.subplot(222)
        time_stamp = 5
        for i, position_data in enumerate(position_results):
            plt.plot(range(len(position_data.x_axis)), position_data.x_axis, color = COLORSET[i], label = LABEL[i] )
        plt.xlabel("time step")
        plt.ylabel("X axis")
        plt.subplot(223)
        for i, position_data in enumerate(position_results):
            plt.plot(range(len(position_data.y_axis)), position_data.y_axis, color = COLORSET[i], label = LABEL[i])
        plt.xlabel("time step")
        plt.ylabel("Y axis")
        plt.subplot(224)
        for i, position_data in enumerate(position_results):
            plt.plot(range(len(position_data.z_axis)), position_data.z_axis, color = COLORSET[i], label = LABEL[i])
        plt.xlabel("time step")
        plt.ylabel("Z_axis")

        plt.legend(bbox_to_anchor = (-0.5, 2.05))

        self.fig = plt.gcf()
        self.fig.savefig(self.folder_name +"/Results_on_timestep.png")
