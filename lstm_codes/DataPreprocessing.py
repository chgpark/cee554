from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np
from random import shuffle
class DataManager:
    def __init__(self, dir, sequence_length, num_uwb):
        self.dir = dir
        self.seq_length = sequence_length
        self.num_uwb = num_uwb
        # scaler saves min / max value of data
       ##########Usage##########
        # scalar = MinMaxScaler()
        # scalar.fit(data)
        # a = scalar.transform(data)
        # b = scalar.inverse_transform(a)

        self.scaler = MinMaxScaler()
        self.scaler_for_prediction = MinMaxScaler()

    def fitDataForMinMaxScaler(self):
        xy = np.loadtxt(self.dir, delimiter=',')
        # if (not prediction):
        self.scaler.fit(xy)
        '''Below one is essential for test!!
            The reason why its range is self.num_uwb: self.num_uwb*3 is to able to operate wheter gt is position or pose.
        '''
        self.scaler_for_prediction.fit(xy[:, self.num_uwb:-self.num_uwb*3])

    def set_range_data_for_4_uwb(self):
        xy = np.loadtxt(self.dir, delimiter=',')
        xy = self.scaler.transform(xy)

        x = xy[:,:self.num_uwb]

        d0_data =[]
        d1_data =[]
        d2_data =[]
        d3_data =[]

        for i in range(self.seq_length - 1):
            for j in range(self.num_uwb):
                _x = []
                for k in range(i+1):
                    _x.append([x[k, j]])

                _x = _x + [[0]]*(self.seq_length - i - 1)

                if j == 0:
                    d0_data.append(_x)
                elif j == 1:
                    d1_data.append(_x)
                elif j == 2:
                    d2_data.append(_x)
                elif j == 3:
                    d3_data.append(_x)


        for i in range(len(x) - self.seq_length + 1):
            for j in range(self.num_uwb):
                _x = []
                for k in range(self.seq_length):
                    _x.append([x[i+k, j]])

                if j == 0:
                    d0_data.append(_x)
                elif j == 1:
                    d1_data.append(_x)
                elif j == 2:
                    d2_data.append(_x)
                elif j == 3:
                    d3_data.append(_x)

        d0_data = np.array(d0_data)
        d1_data = np.array(d1_data)
        d2_data = np.array(d2_data)
        d3_data = np.array(d3_data)

        return d0_data, d1_data, d2_data, d3_data

    def set_gt_data(self):
        xy = np.loadtxt(self.dir, delimiter=',')

        xy = self.scaler.transform(xy)

        robot_pose = xy[:,self.num_uwb:(-1)*self.num_uwb*3]  # Close as label
        relative_cartesian_position = xy[:, (-1)*self.num_uwb*3:]

        robot_pose_data =[]
        relative_position_anchor_data = []

        for i in range(len(robot_pose)):
            robot_pose_data.append(robot_pose[i])
            relative_position_anchor_data.append(relative_cartesian_position[i])

        robot_pose_data = np.array(robot_pose_data)
        relative_position_anchor_data = np.array(relative_position_anchor_data)

        return robot_pose_data, relative_position_anchor_data

    def suffle_array_in_the_same_order(self,*argv):
        index = np.arange((argv[0].shape[0]))
        shuffle(index)
        output_argv =[]

        for arg in argv:
            shuffled_arg = arg[index]
            output_argv.append(shuffled_arg)

        return output_argv
    def set_test_data(self, isRaw = False):
        #set depends on sequence length
        #Just do MinMax Scaler to whole data

        xy = np.loadtxt(self.dir, delimiter=',')
        if (not isRaw):
            xy = self.scaler.transform(xy)
        x = xy[:, :4]
        y = xy[:, 4:]  # Close as label
        # print (type(x))
        # print (type(y))
        X_data=[]
        Y_data=[]
        for i in range(int(len(y)/self.seq_length)):
            idx = i*self.seq_length
            _x = x[idx:idx+self.seq_length]
            _y = y[idx:idx+self.seq_length]
            X_data.append(_x)
            Y_data.append(_y)
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)
        #return numpy array
        return X_data, Y_data

    def inverse_transform_by_train_data(self, prediction):
        # scaler for inverse transform of prediction
        self.inverse_transformed_sequence = self.scaler_for_prediction.inverse_transform(list(prediction[0]))

    def write_file_data(self, out_dir):
        result_file = open(out_dir, 'w', encoding='utf-8', newline='')

        wr = csv.writer(result_file)
        for i in self.inverse_transformed_sequence:
            wr.writerow(i)

        result_file.close()



#Below Line : Extract colums that we want to extract#
#
if __name__ == '__main__':
    file_name = 'inputs/3D_path_poly.csv'
    file  = np.loadtxt(file_name, delimiter= ',')
    seq_length = 10

    num_uwb = 4
    data_parser = DataManager(file_name,seq_length, num_uwb)


    data_parser.fitDataForMinMaxScaler()
    x = np.array([[1,3],[2,7],[3, 10], [4, 1313],[5,1]])
    y = np.array(['a','b','c','d', 'e'])
    z = np.array([111,222,333,444,555])
    # x,y,z = data_parser.suffle_array_in_the_same_order(x, y, z)

    gt, _ = data_parser.set_gt_data()
    print (gt)
    data_parser.inverse_transform_by_train_data(list(gt))
    print ()
    # gt2 = []
    # for j in gt:
    #     gt2.append(list(j))
    # gt2 = tuple(gt2)

    # b = data_parser.scaler_for_prediction.inverse_transform(gt2)

    # total_length = 0
    # data_parser.write_file_data("hello.csv", b)

    # with open('results/test_diagonal_gt.csv' ,'w') as fp:
    #     for i in range(int( total_length/seq_length) ):
    #         np.savetxt(fp,Y_test[i],delimiter=",")
#


