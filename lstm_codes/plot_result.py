import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np
import os
from scipy import interpolate
from scipy.interpolate import spline
p =argparse.ArgumentParser()
p.add_argument('--save_dir', type=str, default="results/RiTA/graphs/")
# p.add_argument('--gt_dir', type=str, default="inputs/test_data_arbitrary_square_uwb_2D_e10.csv")
p.add_argument('--gt_dir', type=str, default="inputs/test_data_diagonal_curve2D.csv")
#In case of test 1
# p.add_argument('--bidirectional_LSTM_csv', type=str, default="results/RiTA/bidirectional_wo_fcn_well_trained.csv")
# p.add_argument('--stacked_bi_LSTM_csv', type=str, default="results/RiTA/stack_bi_2.csv")
# p.add_argument('--unidirectional_LSTM_csv', type=str, default= "results/RiTA/unidirectional_wo_fcn.csv")
# p.add_argument('--gru_csv', type=str, default= "results/RiTA/gru.csv")

#In case of test 2
p.add_argument('--bidirectional_LSTM_csv', type=str, default="results/RiTA/bi_lstm_to_curve_test.csv")
p.add_argument('--stacked_bi_LSTM_csv', type=str, default="results/RiTA/stack_lstm_2.csv")
p.add_argument('--unidirectional_LSTM_csv', type=str, default= "results/RiTA/uni_lstm_to_curve_test.csv")
p.add_argument('--gru_csv', type=str, default= "results/RiTA/gru_to_curve_test.csv")

p.add_argument('--trilateration_csv', type=str, default="results/RiTA/trilateration.csv")
p.add_argument('--save_MSE_name', type=str, default="Distance_error_result__test1.png")
p.add_argument('--save_error_percent_name', type=str, default="test_stack.png")
p.add_argument('--save_trajectory_name', type=str, default="Test_trajectory11_legend.png") #""Trajectory_result_refined_interval_10_smoothed_test_stack.png")
p.add_argument('--data_interval', type=int, default= 21)

args = p.parse_args()
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
LINE = ['-.', ':', '--', '-']
LABEL = ['LSTM', 'GRU', 'Bi-LSTM', 'Stacked Bi-LSTM']

SMOOTHNESS = 200



class Visualization:
    def __init__(self, args):
        self.args = args
        self.folder_name = "results/"
        if not os.path.isdir(self.folder_name):
            os.mkdir(self.folder_name)
        self.setGT()
        self.color_set = COLORSET
        self.label = LABEL
    def setGT(self):
        gt_xy = np.loadtxt(args.gt_dir, delimiter=',')
        #x_array: gt_xy[:,0]
        #y_array: gt_xy[:,1]
        self.gt_xy = gt_xy[4:,4:]

    def _calDistanceError(self, predicted_result_dir):
        predicted_xy = np.loadtxt(predicted_result_dir, delimiter=',')
        # gt_xy = np.random.randint(3,size = (4,2))
        # predicted_xy = np.random.randint(3, size = (4,2))
        dx_dy_array = self.gt_xy - predicted_xy

        distance_square = np.square(dx_dy_array[:,0]) + np.square(dx_dy_array[:,1])
        MSE = np.sum(distance_square)/distance_square.shape
        RMSE = np.sqrt(MSE)
        print ("RMSE: " + str(RMSE*100) + " cm")

        return np.sqrt(distance_square)

    def plotDistanceError(self, *target_files_csv):
        saved_file_name = self.args.save_MSE_name
        plot_title = "Distance Error"
        plt.title(plot_title)
        # plt.rcParams['Figure.figsize'] = (14, 3)
        plt.figure(figsize=(7,4.326))
        for i, csv in enumerate(target_files_csv):

            distance_error = self._calDistanceError(csv)
            distance_error = distance_error*100
            distance_error_refined = self.getRefinedData(distance_error, 30)

            x_axis = range(distance_error.shape[0])
            x_axis_refined = self.getRefinedData(x_axis, 30)

            # x_axis_refined, distance_error_refined = self.getSmoothedData(x_axis_refined, distance_error_refined)
            # x_axis = self.getRefinedData( x_axis, self.args.data_interval)
            # distance_error = self.getRefinedData( distance_error, self.args.data_interval)
            #marker o x + * s:square d:diamond p:five_pointed star


            # plt.plot(x_axis, distance_error, color= SOFT_COLORSET[i], #marker = marker,
            #                 linestyle = linestyle,label = self.label[i])

            plt.plot(x_axis_refined, distance_error_refined, color= self.color_set[i], #marker = marker,
                            linestyle = LINE[i],label = self.label[i])
            # plt.scatter(x_for_marker, distance_error_for_marker, color= self.color_set[i], marker = marker,
            #                 linestyle = linestyle) #,label = self.label[i])


        plt.legend()
        plt.grid(True)
        plt.xlim(0,1500)
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        plt.ylim(0.0,40)
        plt.xlabel("Time Step t")
        plt.ylabel("Distance Error [cm]")
        fig = plt.gcf()
        plt.show()
        fig.savefig(saved_file_name)
        print ("Done")
    def getSmoothedData(self,x_data, y_data):
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        tck, u = interpolate.splprep([x_data, y_data], s=0)
        unew = np.arange(0, 1.01, 0.01)
        out = interpolate.splev(unew, tck)

        smoothed_x = out[0].tolist()
        smoothed_y = out[1].tolist()

        return smoothed_x, smoothed_y

    def getRefinedData(self, data, interval):
        count = 0
        refined_data = []
        for datum in data:
            if count%interval == 0 :
                refined_data.append(datum)
            count += 1
        return refined_data
    def plotErrorPercent(self,*target_files_csv):
        max_value = 0

        for i, csv in enumerate(target_files_csv):
            predicted_xy = np.loadtxt(csv, delimiter = ',')
            predicted_x = predicted_xy[:,0]
            predicted_y = predicted_xy[:,1]
        saved_file_name = self.args.save_error_percent_name
        # plot_title = "CDF of Distance Errors"
        # plt.title(plot_title)

        for i, csv in enumerate(target_files_csv):
            distance_error = self._calDistanceError(csv)
            x_axis = range(distance_error.shape[0])
            interval = 1000
            x_axis =np.linspace(0, np.max(distance_error), interval)
            y = [0]*interval
            for error in distance_error:
                min_residual = 100
                idx = 0
                for j, x in enumerate(x_axis):
                    residual = abs(x - error)
                    if (residual < min_residual):
                        idx = j
                        min_residual = residual
                y[idx] += 1

            y_axis =[0]*interval
            for s in range(interval):
                CDF_y_value = 0
                for t in range(interval):
                    if t <= s:
                        CDF_y_value += y[t]
                        y_axis[s] = CDF_y_value*100/distance_error.shape[0]

            x_axis = x_axis*100
            plt.plot(x_axis, y_axis, color=self.color_set[i],  # marker= marker,
                     linestyle=LINE[i], label=self.label[i])

        plt.grid(True)
        plt.xlim(0.0,40.0)
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        plt.ylim(0.0,100.0)
        plt.legend()
        # plt.xlim(-0.5,1.5)
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.ylim(-0.5,1.5)
        plt.xlabel("Distance Error [cm]")
        plt.ylabel("Percentage [%]")
        fig = plt.gcf()
        plt.show()
        fig.savefig(saved_file_name)
        print("Done")


    def plot2DTrajectory(self, *target_files_csv):
        saved_file_name = self.args.save_trajectory_name
        plot_title = "Trajectory"
        # plt.title(plot_title)
        gt_x = self.gt_xy[:,0]
        gt_y = self.gt_xy[:,1]

        plt.figure(figsize=(8, 6))
        plt.plot(gt_x, gt_y,'k',linestyle='--' , label = 'GT')

        for i, csv in enumerate(target_files_csv):
            predicted_xy = np.loadtxt(csv, delimiter = ',')
            predicted_x = predicted_xy[:,0]
            predicted_y = predicted_xy[:,1]

            predicted_x = self.getRefinedData( predicted_x, self.args.data_interval)
            predicted_y = self.getRefinedData( predicted_y, self.args.data_interval)

            predicted_x, predicted_y = self.getSmoothedData(predicted_x, predicted_y)
            #marker o x + * s:square d:diamond p:five_pointed star

            plt.plot(predicted_x, predicted_y, color = self.color_set[i], #marker= marker,
                            linestyle = LINE[i],label = self.label[i])

        plt.legend()

        # plt.legend(bbox_to_anchor=(1, 1),
        #            bbox_transform=plt.gcf().transFigure)
        # plt.xlim(-0.5,1.5)
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.xticks(np.linspace(-0.5,1.5,10, endpoint =True))
        # plt.ylim(-0.5,1.5)
        plt.xlabel("X Axis [m]")
        plt.ylabel("Y Axis [m]")
        fig = plt.gcf()
        plt.show()
        fig.savefig(saved_file_name)
        print ("Done")

    def drawResult3D(self, X_list, Y_list, Z_list):

        self.fig = plt.figure()
        # plt.subplot(221)
        # self.ax1 = self.fig.gca(projection = '3d') #add_subplot(111, projection='3d')
        # self.ax1.scatter(X_list, Y_list, Z_list)
        plt.subplot(221)
        plt.scatter(X_list, Y_list)
        plt.xlabel("X_axis")
        plt.ylabel("Y_axis")
        plt.subplot(222)
        plt.scatter(Z_list, Y_list)
        plt.xlabel("Y_axis")
        plt.ylabel("Z_axis")
        plt.subplot(223)
        plt.scatter(X_list, Z_list)
        plt.xlabel("X_axis")
        plt.ylabel("Z_axis")
        self.fig = plt.gcf()
        self.fig.savefig(self.folder_name +"/Results.png")

if __name__ == "__main__":
    viz = Visualization(args)
    # viz.plotDistanceError(args.unidirectional_LSTM_csv, args.gru_csv, args.bidirectional_LSTM_csv, args.stacked_bi_LSTM_csv)
    # viz.plotErrorPercent(args.unidirectional_LSTM_csv, args.gru_csv, args.bidirectional_LSTM_csv, args.stacked_bi_LSTM_csv)
    viz.plot2DTrajectory(args.unidirectional_LSTM_csv, args.gru_csv, args.bidirectional_LSTM_csv, args.stacked_bi_LSTM_csv)

