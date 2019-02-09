from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np
from random import shuffle
import os
class DataManager:
    def __init__(self, args):
        self.dir = args.train_data
        self.seq_length = args.sequence_length
        self.num_uwb = args.num_uwb
        self.grid = args.grid_size
        # scaler saves min / max value of data
       ##########Usage##########
        # scalar = MinMaxScaler()
        # scalar.fit(data)
        # a = scalar.transform(data)
        # b = scalar.inverse_transform(a)

        self.scaler = MinMaxScaler()
        self.scaler_for_prediction = MinMaxScaler()
    def set_dir(self, dir):
        self.dir = dir

   ##################################################
                 #Preprocessing parts
   ##################################################
    def set_all_target_data_list(self, generating_grid):
        train_file_list = os.listdir(self.dir)

        self.train_files_dir= []
        for train_file in train_file_list:
           self.train_files_dir.append(os.path.join(self.dir, train_file))

        if generating_grid:
            self.train_data_list = []
            for train_file_dir in self.train_files_dir:
                print ("Load " + train_file_dir)
                train_data = np.loadtxt(train_file_dir, delimiter=',')
                position_in_train_data = train_data[:, self.num_uwb : self.num_uwb + 3]
                rounded_position = np.round(position_in_train_data/self.grid) * self.grid
                rounded_train_data = np.concatenate((train_data[:,:self.num_uwb], rounded_position), axis = 1)
                self.train_data_list.append(rounded_train_data)

        else:
            self.train_data_list = []
            for train_file_dir in self.train_files_dir:
                print ("Load " + train_file_dir)
                train_data = np.loadtxt(train_file_dir, delimiter=',')
                self.train_data_list.append(train_data[:,:11])

    def set_val_data(self, val_data_dir):
        self.train_data_list = []
        val_data = np.loadtxt(val_data_dir, delimiter=',')
        self.train_data_list.append(val_data[:,:11])

    def fitDataForMinMaxScaler(self, generating_grid = True):
        self.set_all_target_data_list(generating_grid)

        xy = self.train_data_list[0].copy()
        if (len(self.train_data_list) > 1):
            for train_data in self.train_data_list[1:]:
                xy = np.concatenate((xy, train_data), axis = 0)
        # if (not prediction):
        self.scaler.fit(xy)
        '''
            Below one is essential for test!!
            The reason why its range is self.num_uwb: self.num_uwb*3 is to able to operate wheter gt is position or pose.
        '''
        self.scaler_for_prediction.fit(xy[:, self.num_uwb:self.num_uwb+3])
        del xy

    def transform_all_data(self):
        train_data_list = self.train_data_list.copy()
        for i, train_data in enumerate(train_data_list):
            xy = self.scaler.transform(train_data)
            self.train_data_list[i] = xy


    ##################################################
                 #Setting train data parts
    ##################################################


    def set_range_data(self):
        '''For non-multimodal!'''
        self.X_data = []
        for train_data in self.train_data_list:
            x = train_data[:,:self.num_uwb]

            for i in range(len(x) - self.seq_length + 1):
                _x = x[i:i+self.seq_length]
                self.X_data.append(_x)

        self.X_data = np.array(self.X_data)

    def set_range_data_for_4multimodal(self):
        self.d0_data =[]
        self.d1_data =[]
        self.d2_data =[]
        self.d3_data =[]

        for train_data in self.train_data_list:
            xy = train_data
            xy = self.scaler.transform(xy)
            x = xy[:,:self.num_uwb]

            for i in range(len(x) - self.seq_length + 1):
                for j in range(self.num_uwb):
                    _x = []
                    for k in range(self.seq_length):
                        _x.append([x[i+k, j]])

                    if j == 0:
                        self.d0_data.append(_x)
                    elif j == 1:
                        self.d1_data.append(_x)
                    elif j == 2:
                        self.d2_data.append(_x)
                    elif j == 3:
                        self.d3_data.append(_x)


        self.d0_data = np.array(self.d0_data)
        self.d1_data = np.array(self.d1_data)
        self.d2_data = np.array(self.d2_data)
        self.d3_data = np.array(self.d3_data)

    def set_range_data_for_8multimodal(self):
        self.d0_data =[]
        self.d1_data =[]
        self.d2_data =[]
        self.d3_data =[]
        self.d4_data =[]
        self.d5_data =[]
        self.d6_data =[]
        self.d7_data =[]

        for train_data in self.train_data_list:
            x = train_data[:,:self.num_uwb]

            for i in range(len(x) - self.seq_length + 1):
                for j in range(self.num_uwb):
                    _x = []
                    for k in range(self.seq_length):
                        _x.append([x[i+k, j]])

                    if j == 0:
                        self.d0_data.append(_x)
                    elif j == 1:
                        self.d1_data.append(_x)
                    elif j == 2:
                        self.d2_data.append(_x)
                    elif j == 3:
                        self.d3_data.append(_x)
                    elif j == 4:
                        self.d4_data.append(_x)
                    elif j == 5:
                        self.d5_data.append(_x)
                    elif j == 6:
                        self.d6_data.append(_x)
                    elif j == 7:
                        self.d7_data.append(_x)

        self.d0_data = np.array(self.d0_data)
        self.d1_data = np.array(self.d1_data)
        self.d2_data = np.array(self.d2_data)
        self.d3_data = np.array(self.d3_data)
        self.d4_data = np.array(self.d4_data)
        self.d5_data = np.array(self.d5_data)
        self.d6_data = np.array(self.d6_data)
        self.d7_data = np.array(self.d7_data)

    def set_gt_data(self):
        self.position_data =[]
        self.quaternion_data = []

        for train_data in self.train_data_list:
            robot_position = train_data[:, self.num_uwb: self.num_uwb + 3]  # Close as label
            robot_quaternion = train_data[:, self.num_uwb+3:]

            for i in range(self.seq_length-1, len(robot_position)):
                self.position_data.append(robot_position[i])
                self.quaternion_data.append(robot_quaternion[i])
        self.position_data = np.array(self.position_data)
        self.quaternion_data = np.array(self.quaternion_data)


    def set_gt_data_for_all_sequences(self):
            self.position_data =[]
            self.quaternion_data = []

            for train_data in self.train_data_list:
                robot_position = train_data[:,self.num_uwb: self.num_uwb + 3]  # Close as label
                robot_quaternion = train_data[:,self.num_uwb+3:]

                for i in range(len(robot_position) - self.seq_length +1):
                    _position = []
                    _quaternion = []
                    for j in range(self.seq_length):
                        _position.append(robot_position[i+j,:])
                        # _quaternion.append(robot_quaternion[:,i+j])
                    self.position_data.append(_position)
                    # self.quaternion_data.append(_quaternion)

            self.position_data = np.array(self.position_data)
            self.quaternion_data = np.array(self.quaternion_data)

    def set_data_for_non_multimodal_all_sequences(self):
        self.set_range_data()
        self.set_gt_data_for_all_sequences()

    ##################################################
                    #For FC layer
    ##################################################
    def set_range_data_for_fc_layer(self):
        '''For non-multimodal!'''
        self.X_data = []
        for train_data in self.train_data_list:
            x = train_data[:,:self.num_uwb]

            for i in range(len(x)):
                _x = x[i]
                self.X_data.append(_x)

        self.X_data = np.array(self.X_data)

    def set_gt_data_for_fc_layer(self):
            self.position_data =[]
            self.quaternion_data = []

            for train_data in self.train_data_list:
                robot_position = train_data[:, self.num_uwb: self.num_uwb + 3]  # Close as label
                robot_quaternion = train_data[:, self.num_uwb+3:]

                for i in range(len(robot_position)):
                    self.position_data.append(robot_position[i])
                    self.quaternion_data.append(robot_quaternion[i])
            self.position_data = np.array(self.position_data)
            self.quaternion_data = np.array(self.quaternion_data)

    def set_data_for_fc_layer(self):
        self.set_range_data_for_fc_layer()
        self.set_gt_data_for_fc_layer()

    def set_data_for_4multimodal(self):
        self.set_range_data_for_4multimodal()
        self.set_gt_data()

    def set_data_for_8multimodal(self):
        self.set_range_data_for_8multimodal()
        self.set_gt_data()


    def set_data_for_8multimodal_all_sequences(self):
        self.set_range_data_for_8multimodal()
        self.set_gt_data_for_all_sequences()

    ##################################################
                    #Getting data
    ##################################################

    def get_range_data_for_4multimodal(self):
        return self.d0_data, self.d1_data , self.d2_data, self.d3_data

    def get_range_data_for_8multimodal(self):
        return self.d0_data, self.d1_data , self.d2_data, self.d3_data, self.d4_data, self.d5_data, self.d6_data, self.d7_data

    def get_range_data_for_nonmultimodal(self):
        return self.X_data

    def get_gt_data(self):
        return self.position_data, self.quaternion_data

    def suffle_array_in_the_same_order(self,*argv):
        index = np.arange((argv[0].shape[0]))
        shuffle(index)
        output_argv =[]

        for arg in argv:
            shuffled_arg = arg[index]
            output_argv.append(shuffled_arg)

        return output_argv

    def inverse_transform_by_train_data(self, prediction):
        # scaler for inverse transform of prediction
        self.inverse_transformed_sequence = self.scaler_for_prediction.inverse_transform(list(prediction))

    def write_file_data(self, out_dir):
        result_file = open(out_dir, 'w', encoding='utf-8', newline='')

        wr = csv.writer(result_file)
        for i in self.inverse_transformed_sequence:
            wr.writerow(i)

        result_file.close()


#Below Line : Extract colums that we want to extract#
#
if __name__ == '__main__':
    file_name = '/home/shapelim/RONet/train_debug/'
    file_name2 = '/home/shapelim/RONet/train_debug/np_data_2.csv'
    test_name = 'inputs/np_test_data_2.csv'
    aa = np.loadtxt(file_name2, delimiter= ',')

    # scalar = MinMaxScaler()
    # scalar_prediction = MinMaxScaler()
    # scalar.fit(aa)
    #
    # position =  aa[:,8:]
    # scalar_prediction.fit(position)
    # dd = scalar_prediction.transform(position)
    # dd =scalar_prediction.inverse_transform(dd)
    # a = scalar.transform(aa)
    # b = scalar.inverse_transform(a)
    # print (b)
    # print (dd)
    # print (aa)
    data_parser = DataManager(file_name, 5, 8)
    data_parser.fitDataForMinMaxScaler()
    # #
    data_parser.transform_all_data()
    c = data_parser.scaler.inverse_transform(data_parser.train_data_list[0])
    # print (c)
    data_parser.set_data_for_8multimodal()
    d0_data, d1_data, d2_data, d3_data, d4_data, d5_data, d6_data, d7_data = data_parser.get_range_data_for_8multimodal()
    robot_position_gt, robot_quaternion_gt = data_parser.get_gt_data()

    data_parser.set_val_data(test_name)
    data_parser.transform_all_data()
    data_parser.set_data_for_8multimodal()
    d0_data, d1_data, d2_data, d3_data, d4_data, d5_data, d6_data, d7_data = data_parser.get_range_data_for_8multimodal()
    val_robot_position_gt, val_robot_quaternion_gt = data_parser.get_gt_data()


    data_parser.inverse_transform_by_train_data(val_robot_position_gt)
    a = data_parser.inverse_transformed_sequence
    print (data_parser.inverse_transformed_sequence)
    # #
    # #
    # # data_parser.set_val_data(test_name)
    # # data_parser.set_train_data_for_8multimodal()
    # # data_parser.transform_all_data()
    # # gt, _ = data_parser.get_gt_data()
    # #
    # # print (gt)

    # print (a)
    # b = data_parser.scaler.inverse_transform(data_parser.train_data_list[0])
    # print (b)
