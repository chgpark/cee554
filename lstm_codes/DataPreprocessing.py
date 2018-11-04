from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np
from random import shuffle
import os
class DataManager:
    def __init__(self, train_files_dir, sequence_length, num_uwb):
        self.dir = train_files_dir
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
    def set_dir(self, dir):
        self.dir = dir

   ##################################################
                 #Preprocessing parts
   ##################################################
    def set_all_train_data_list(self):
        train_file_list = os.listdir(self.dir)

        self.train_files_dir= []
        for train_file in train_file_list:
           self.train_files_dir.append(os.path.join(self.dir, train_file))

        self.train_data_list = []
        for train_file_dir in self.train_files_dir:
            a = np.loadtxt(train_file_dir, delimiter=',')
            self.train_data_list.append(a)

    def set_val_data(self, val_data_dir):
        self.train_data_list = []
        val_data = np.loadtxt(val_data_dir, delimiter=',')
        self.train_data_list.append(val_data)

    def fitDataForMinMaxScaler(self):
        self.set_all_train_data_list()

        xy = self.train_data_list[0].copy()
        if (len(self.train_data_list) > 1):
            for train_data in self.train_data_list[1:]:
                xy = np.concatenate((xy, train_data), axis = 0)
        # if (not prediction):
        self.scaler.fit(xy)
        '''Below one is essential for test!!
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
            robot_position = train_data[:,self.num_uwb: self.num_uwb + 3]  # Close as label
            robot_quaternion = train_data[:,self.num_uwb+3:]


            for i in range(self.seq_length-1, len(robot_position)):
                self.position_data.append(robot_position[i])
                self.quaternion_data.append(robot_quaternion[i])
        self.position_data = np.array(self.position_data)
        self.quaternion_data = np.array(self.quaternion_data)

    def set_train_data_for_non_multimodal(self):
        self.set_range_data()
        self.set_gt_data()

    def set_train_data_for_4multimodal(self):
        self.set_range_data_for_4multimodal()
        self.set_gt_data()

    def set_train_data_for_8multimodal(self):
        self.set_range_data_for_8multimodal()
        self.set_gt_data()

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
    file_name = '/home/shapelim/RONet/train_test/'

    b = os.listdir(file_name)
    total_np = np.array([[None]*19])
    for i, file in enumerate(b):
        adsd = os.path.join(file_name, file)
        if (i == 0):

            c = np.loadtxt(adsd, delimiter=',')

        elif (i == 1):
            d = np.loadtxt(adsd, delimiter=',')

        zzzz= np.loadtxt(adsd, delimiter = ',')
        total_np = np.concatenate((total_np, zzzz), axis = 0)
    print (len(total_np), total_np[0])
    total_np = total_np[1:]

    print (total_np[0])
    # print (c[0], len(c[0]))
    # print (len(c))
    # print (c)
    # print ("heelo")
    # e = np.concatenate((c,d), axis = 0)
    # print (len(c), len(d), len(e))
            # total_np = np.concatenate([total_np, c], axis = 0)

        # print (np, np.shape)

        # print (adsd)

    # file  = np.loadtxt(file_name, delimiter= ',')



    seq_length = 10
    num_uwb = 4
    # data_parser = DataManager(file_name, seq_length, num_uwb)
    #
    # data_parser.fitDataForMinMaxScaler()
    # x = np.array([[1,3],[2,7],[3, 10], [4, 1313],[5,1]])
    # y = np.array(['a','b','c','d', 'e'])
    # z = np.array([111,222,333,444,555])
    # x,y,z = data_parser.suffle_array_in_the_same_order(x, y, z)

    # gt, _ = data_parser.set_gt_data()
    # print (gt)
    # data_parser.inverse_transform_by_train_data(list(gt))
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


