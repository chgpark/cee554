from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np
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
        # else:
        self.scaler_for_prediction.fit(xy[:,4:])

    def set_data(self):
        xy = np.loadtxt(self.dir, delimiter=',')

        # xy = self.scaler.transform(xy)

        x = xy[:,:self.num_uwb]
        robot_pose = xy[:,self.num_uwb:(-1)*self.num_uwb*3]  # Close as label
        relative_cartesian_position = xy[:, (-1)*self.num_uwb*3:]

        X_data =[]
        robot_pose_data =[]
        relative_position_anchor_data = []

        for i in range(self.seq_length - 1):
            range_list = []
            for j in range(self.num_uwb):
                _x = []
                for k in range(i+1):
                    _x.append(x[k, j])

                _x = _x + [0]*(self.seq_length - i - 1)
                range_list.append(_x)

            X_data.append(range_list)
            robot_pose_data.append(robot_pose[i])
            relative_position_anchor_data.append(relative_cartesian_position[i])

        for i in range(len(x) - self.seq_length + 1):
            range_list = []
            for j in range(self.num_uwb):
                _x = x[i:i+self.seq_length, j]
                range_list.append(_x)

            X_data.append(range_list)
            robot_pose_data.append(robot_pose[i + self.seq_length - 1])
            relative_position_anchor_data.append(relative_cartesian_position[i + self.seq_length - 1])

        X_data = np.array(X_data)
        robot_pose_data = np.array(robot_pose_data)
        relative_position_anchor_data = np.array(relative_position_anchor_data)

        X_data, robot_pose_data, relative_position_anchor_data = self.suffle_array_in_the_same_order(X_data, robot_pose_data, relative_position_anchor_data)

        return X_data, robot_pose_data, relative_position_anchor_data

    def suffle_array_in_the_same_order(self,x_data, y_data, z_data):
        shuffle_index = np.arange(x_data.shape[0])
        x_data = x_data[shuffle_index]
        y_data = y_data[shuffle_index]
        z_data = z_data[shuffle_index]

        return x_data, y_data, z_data
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


    def write_file_data(self, out_dir, prediction):
        result_file = open(out_dir, 'w', encoding='utf-8', newline='')
        wr = csv.writer(result_file)

        for sequence_list in prediction[0]: # bc shape of prediction is "[" [[[hidden_size]*sequence_length], ... ] "]"
            # np_sequence = np.array(sequence_list, dtype=np.float32)
            #
            # # scaler for inverse transform of prediction
            # transformed_sequence = self.scaler_for_prediction.inverse_transform(np_sequence)
            # for i in transformed_sequence:
            #     wr.writerow([i[0], i[1]])


            # scaler for inverse transform of prediction
            transformed_sequence = self.scaler_for_prediction.inverse_transform([sequence_list])
            for i in transformed_sequence:
                wr.writerow([i[0], i[1]])

        result_file.close()



#Below Line : Extract colums that we want to extract#
#
if __name__ == '__main__':
    file_name = 'train_data.csv'
    seq_length = 10
    num_uwb = 4
    data_parser = DataManager(file_name,seq_length, num_uwb)
    data_parser.fitDataForMinMaxScaler()
    X_test, Y_test = data_parser.set_data()

    total_length = 0
    with open(file_name) as f:
        for num_line, l in enumerate(f):  # For large data, enumerate should be used!
            pass
        total_length = num_line
    total_length +=1
    # with open('results/test_diagonal_gt.csv' ,'w') as fp:
    #     for i in range(int( total_length/seq_length) ):
    #         np.savetxt(fp,Y_test[i],delimiter=",")
#


