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
        self.input_size = args.num_uwb
        self.train_dir = args.train_data
        self.val_dir = args.val_data
        if args.mode == 'test':
            self.test_dir = args.test_data
        # scaler saves min / max value of data
       ##########Usage##########
        # scalar = MinMaxScaler()
        # scalar.fit(data)
        # a = scalar.transform(data)
        # b = scalar.inverse_transform(a)

        self.train_files_dir = []
        self.val_files_dir = []
        
        self.scaler = MinMaxScaler()
        self.scaler_for_prediction = MinMaxScaler()
    def set_dir(self, dir):
        self.dir = dir

   ##################################################
                 #Preprocessing parts
   ##################################################
    def set_all_target_data_list(self):
        train_file_list = os.listdir(self.train_dir)
        val_file_list = os.listdir(self.val_dir)

        for train_file in train_file_list:
            self.train_files_dir.append(os.path.join(self.train_dir, train_file))

        for val_file in val_file_list:
            self.val_files_dir.append(os.path.join(self.val_dir, val_file))

    def fit_train_data(self):
        for train_file_dir in self.train_files_dir:
            print("Fitting " + train_file_dir)
            train_data = np.loadtxt(train_file_dir, delimiter=',')
            uwb_data = train_data[:, :self.input_size]
            self.scaler.partial_fit(uwb_data)

            position_data = train_data[:, self.input_size:self.input_size + 2]
            self.scaler_for_prediction.partial_fit(position_data)

    def fit_val_data(self):
        for val_file_dir in self.val_files_dir:
            print("Fitting " + val_file_dir)
            val_data = np.loadtxt(val_file_dir, delimiter=',')
            uwb_data = val_data[:, :self.input_size]
            self.scaler.partial_fit(uwb_data)

            position_data = val_data[:, self.input_size:self.input_size + 2]
            self.scaler_for_prediction.partial_fit(position_data)

    def fit_all_data(self):
        print("On fitting all data...")
        self.set_all_target_data_list()
        self.fit_train_data()
        self.fit_val_data()
    ##################################################
                 #Setting train data parts
    ##################################################

    def set_train_data(self):
        '''For non-multimodal!'''
        self.X_data = []
        self.position_data =[]

        for train_file_dir in self.train_files_dir:
            train_data = np.loadtxt(train_file_dir , delimiter=',')

            range_data = train_data[:, :self.input_size]
            p_data = train_data[:, self.input_size:self.input_size + 2]

            range_data = self.scaler.transform(range_data)
            p_data = self.scaler_for_prediction.transform(p_data)


            for i in range(len(range_data) - self.seq_length + 1):
                _x = range_data[i:i+self.seq_length]
                _position = p_data[i:i+self.seq_length]
                self.X_data.append(_x)
                self.position_data.append(_position)

        self.X_data = np.array(self.X_data)
        self.position_data = np.array(self.position_data)

    def set_val_data(self):
        '''For non-multimodal!'''
        self.X_data = []
        self.position_data = []

        for val_file_dir in self.val_files_dir:
            val_data = np.loadtxt(val_file_dir , delimiter=',')

            range_data = val_data[:, :self.input_size]
            p_data = val_data[:, self.input_size:self.input_size + 2]

            range_data = self.scaler.transform(range_data)
            p_data = self.scaler_for_prediction.transform(p_data)

            for i in range(len(range_data) - self.seq_length + 1):
                _x = range_data[i:i+self.seq_length]
                _position = p_data[i:i+self.seq_length]
                self.X_data.append(_x)
                self.position_data.append(_position)

        self.X_data = np.array(self.X_data)
        self.position_data = np.array(self.position_data)
    
    def set_test_data(self):
        self.X_data = []
        self.position_data = []

        test_data = np.loadtxt(self.test_dir, delimiter=',')

        range_data = test_data[:, :self.input_size]
        p_data = test_data[:, self.input_size:self.input_size + 2]

        range_data = self.scaler.transform(range_data)
        p_data = self.scaler_for_prediction.transform(p_data)

        for i in range(len(range_data) - self.seq_length + 1):
            _x = range_data[i:i+self.seq_length]
            _position = p_data[i:i+self.seq_length]
            self.X_data.append(_x)
            self.position_data.append(_position)

        self.X_data = np.array(self.X_data)
        self.position_data = np.array(self.position_data)
        
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

    def set_data_for_non_multimodal_all_sequences(self):
        self.set_range_data()
        self.set_gt_data_for_all_sequences()

    ##################################################
                    #For FC layer
    ##################################################
    def set_train_data_for_fc_layer(self):
        '''For non-multimodal!'''
        self.X_data = []
        self.position_data =[]

        for train_file_dir in self.train_files_dir:
            train_data = np.loadtxt(train_file_dir , delimiter=',')

            range_data = train_data[:, :self.input_size]
            p_data = train_data[:, self.input_size:self.input_size + 2]

            range_data = self.scaler.transform(range_data)
            p_data = self.scaler_for_prediction.transform(p_data)

            for i in range(len(range_data)):
                _x = range_data[i]
                _position = p_data[i]

                self.X_data.append(_x)
                self.position_data.append(_position)

        self.X_data = np.array(self.X_data)
        self.position_data = np.array(self.position_data)

    def set_val_data_for_fc_layer(self):
        '''For non-multimodal!'''
        self.X_data = []
        self.position_data = []

        for val_file_dir in self.val_files_dir:
            val_data = np.loadtxt(val_file_dir , delimiter=',')

            range_data = val_data[:, :self.input_size]
            p_data = val_data[:, self.input_size:self.input_size + 2]

            range_data = self.scaler.transform(range_data)
            p_data = self.scaler_for_prediction.transform(p_data)

            for i in range(len(range_data)):
                _x = range_data[i]
                _position = p_data[i]

                self.X_data.append(_x)
                self.position_data.append(_position)


        self.X_data = np.array(self.X_data)
        self.position_data = np.array(self.position_data)

    def set_test_data_for_fc_layer(self):
        self.X_data = []
        self.position_data = []

        test_data = np.loadtxt(self.test_dir, delimiter=',')

        range_data = test_data[:, :self.input_size]
        p_data = test_data[:, self.input_size:self.input_size + 2]

        range_data = self.scaler.transform(range_data)
        p_data = self.scaler_for_prediction.transform(p_data)

        for i in range(len(range_data)):
            _x = range_data[i]
            _position = p_data[i]

            self.X_data.append(_x)
            self.position_data.append(_position)


        self.X_data = np.array(self.X_data)
        self.position_data = np.array(self.position_data)

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
        return self.position_data

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
