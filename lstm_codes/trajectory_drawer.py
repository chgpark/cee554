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
COLORSET = [(241/255.0, 101/255.0, 65/255.0), (2/255.0, 23/255.0, 157/255.0), (19/255.0, 128/255.0, 20/255.0), (191/255.0, 17/255.0, 46/255.0)]
SOFT_COLORSET = [(241/255.0, 187/255.0, 165/255.0), (174/255.0, 245/255.0, 231/255.0), (115/255.0, 123/255.0, 173/255.0), (232/255.0, 138/255.0, 139/255.0)]
LINE = ['--', '-.', ':', '-.', '-.']
# LABEL = ['LSTM', 'GRU', 'Bi-LSTM', 'Stacked Bi-LSTM']
LABEL = ['Particle Filter', 'MLP', 'Stacked Bi-LSTM', 'Ours']
#marker o x + * s:square d:diamond p:five_pointed star
MARKER = ['p', 'x', '*', 's']
SMOOTHNESS = 30

DATA_INTERVAL = 20
def return_actual_position(x, y, direction):
    real_x = x*0.45
    real_y = y*0.45
    if direction == 'u':
        real_y += 0.058
    elif direction == 'd':
        real_y -= 0.058
    elif direction == 'l':
        real_x -= 0.058
    elif direction == 'r':
        real_x += 0.058

    return real_x, real_y

class Visualization:
    def __init__(self):
        self.color_set =  COLORSET
        self.line = LINE
        self.label = LABEL

    def setGT(self, raw_csv_file):
        gt_xyz = np.loadtxt(raw_csv_file, delimiter=',')
        #x_array: gt_xy[:,0]
        #y_array: gt_xy[:,1]
        self.gt_xyz = gt_xyz[:, -2:]

    def getSmoothedData_2D(self, x_data, y_data):
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        tck, u = interpolate.splprep([x_data, y_data], s=0)
        unew = np.arange(0, 1.01, 0.01)
        out = interpolate.splev(unew, tck)

        smoothed_x = out[0].tolist()
        smoothed_y = out[1].tolist()

        return smoothed_x, smoothed_y

    def set_2D_plot_name(self, name):
        self._2d_plot_name = name

    def getRefinedData(self, data, interval):
        count = 0
        refined_data = []
        for datum in data:
            if count % interval == 0:
                refined_data.append(datum)
            count += 1
        return refined_data

    def draw_2D_trajectory(self, mode, *target_files_csv):
        global DATA_INTERVAL
        saved_file_name = self._2d_plot_name
        plot_title = "Trajectory"
        # plt.title(plot_title)
        gt_x = self.gt_xyz[:, 0]
        gt_y = self.gt_xyz[:, 1]

        plt.figure(figsize=(8, 6))
        plt.plot(gt_x, gt_y,'k',linestyle='-', label = 'GT')

        for i, csv in enumerate(target_files_csv):
            predicted_xy = np.loadtxt(csv, delimiter=',')
            x = []
            y = []
            predicted_x = predicted_xy[:, 0]
            predicted_y = predicted_xy[:, 1]
            for j in range(len(predicted_x)):
                if j % SMOOTHNESS == 0:
                    x.append(predicted_x[i])
                    y.append(predicted_y[i])

            # print(len(x))
            predicted_x = self.getRefinedData( predicted_x, DATA_INTERVAL)
            predicted_y = self.getRefinedData( predicted_y, DATA_INTERVAL)
            #
            # predicted_x, predicted_y = self.getSmoothedData_2D(predicted_x, predicted_y)
            
            #marker o x + * s:square d:diamond p:five_pointed star

            plt.plot(predicted_x, predicted_y, color = self.color_set[i], #marker= MARKER[i],
                            linestyle = LINE[i],label = self.label[i])

        data_list = [(-3, 0, 'u'), (3, -2, 'u'), (6, -4, 'l'), (6, 6, 'd'),
                     (0, 2, 'd'), (-6, -3, 'r'), (1, -5, 'l'), (-5, 5, 'r')]

        print(mode, mode=='3')
        if mode == "3":
            selected_anchor = [data_list[0], data_list[3], data_list[6]]
        elif mode =="5":
            selected_anchor = [data_list[0], data_list[1], data_list[2], data_list[3], data_list[6]]
        elif mode =="8":
            selected_anchor = data_list
        for data in selected_anchor:
            real_x, real_y = return_actual_position(data[0], data[1], data[2])
            plt.scatter(real_x, real_y, c='r', marker='^', s=100)
            # axis_string = '(' + str(round(real_x, 2)) + ', ' + str(round(real_y, 2)) + ')'
            # plt.text(real_x + offset_x, real_y + offset_y, axis_string, fontsize=15)

            # plt.plot(x, y, color = self.color_set[i], #marker= marker,
            #                 linestyle = LINE[i],label = self.label[i])

        # plt.legend()

        plt.grid()
        # plt.legend(bbox_to_anchor=(1, 1),
        #            bbox_transform=plt.gcf().transFigure)
        plt.xlim(-3.0, 3.0)
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        plt.ylim(-2.5, 3.0)
        plt.xlabel("X Axis [m]", fontsize=15)
        plt.ylabel("Y Axis [m]", fontsize=15)
        fig = plt.gcf()
        # plt.show()
        fig.savefig(saved_file_name)
        print ("Done")

    def set_3D_plot_name(self, name):
        self._3D_plot_name = name


    def drawResult3D(self, *target_files_csv):

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111, projection='3d')

        gt_x = self.gt_xyz[:,0]
        gt_y = self.gt_xyz[:,1]
        gt_z = self.gt_xyz[:,2]
        self.ax1.plot(gt_x, gt_y, gt_z, 'k', linestyle = '--', label = 'GT')

        for i, csv in enumerate(target_files_csv):
            predicted_xyz = np.loadtxt(csv, delimiter = ',')
            # print (predicted_xyz)
            predicted_x = predicted_xyz[:,0]
            predicted_y = predicted_xyz[:,1]
            predicted_z = predicted_xyz[:,2]

            # predicted_x = self.getRefinedData( predicted_x, self.args.data_interval)
            # predicted_y = self.getRefinedData( predicted_y, self.args.data_interval)
            # predicted_z = self.getRefinedData( predicted_z, self.args.data_interval)
            #
            # predicted_x, predicted_y, predicted_z = self.getSmoothedData_3D(predicted_x, predicted_y, predicted_z)

            plt.plot(predicted_x, predicted_y, predicted_z, color = self.color_set[i], #marker= marker,
                            linestyle = LINE[i],label = self.label[i])
        plt.legend()

        # self.ax1.scatter(X_list, Y_list, Z_list, c=c)
        self.ax1.set_zlim(0, 1.0)
        self.ax1.set_xlim(-0.75, 1.25)
        self.ax1.set_ylim(-1.0, 1.0)
        self.ax1.set_xlabel('X axis')
        self.ax1.set_ylabel('Y axis')
        self.ax1.set_zlabel('Z axis')
        self.fig = plt.gcf()
        self.fig.savefig(self._3D_plot_name)



if __name__ == "__main__":
    def drawResult3D(X_list, Y_list, Z_list, c):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot(X_list, Y_list, Z_list, c=c )
    def randrange(n, vmin, vmax):
        return (vmax - vmin)*np.random.rand(n) + vmin
    viz = Visualization()
    viz.setGT('RO_test/02-38.csv')

    mode = '8'
    viz.set_2D_plot_name("hi_" + mode+".png")

    abs_dir = '/home/shapelim/git_files/cee554/result_data/'
    PF = abs_dir + "Particle_filter/Particle_filter_" + mode +".csv"
    fc = abs_dir + "fc/fc_" + mode + ".csv"
    stacked_bi = abs_dir + "stacked_bi/stacked_bi_" + mode +".csv"
    RONet = abs_dir + "RONet/RONet_" + mode + ".csv"
    viz.draw_2D_trajectory(mode, PF, fc, stacked_bi, RONet)
    viz.draw_2D_trajectory(mode, PF, RONet)
    # n = 10
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #     xs = randrange(n, 23, 32)
    #     ys = randrange(n, 0, 100)
    #     zs = randrange(n, zlow, zhigh)
    #     drawResult3D(xs, ys, zs, c)
    # plt.show()
    # X = test[:, 4]
    # Y = test[:, 5]
    # Z = test[:, 6]
    # viz.drawResult3D(X, Y, Z)
    # input = np.loadtxt("./inputs/3D_path_spiral.csv", delimiter = ',')
    # input = np.loadtxt("./inputs/3D_path_poly.csv", delimiter = ',')
    # x = input[:, 4]
    # y = input[:, 5]
    # z = input[:, 6]
    # viz.drawResult3D(args.uni) #, args.bi, args.multimodal_uni, args.multimodal_bi)
    # viz.plotDistanceError3D(args.multimodal_bi)
    # viz.plotErrorPercent(args.multimodal_bi)
    # viz.plot2DTrajectory(args.unidirectional_LSTM_csv, args.gru_csv, args.bidirectional_LSTM_csv, args.stacked_bi_LSTM_csv)

