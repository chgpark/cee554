from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np
class DataPreprocessing:
    def __init__(self, dir, sequence_length):
        self.dir = dir
        self.seq_length = sequence_length
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


        # test= xy[0,:4]
        # test1= xy[0,4:]
        # print(test)
        # print(test1)
        xy = self.scaler.transform(xy)

        x = xy[:,:4]
        y = xy[:,4:]  # Close as label
        # print (type(x))
        # print (type(y))
        X_data =[]
        Y_data =[]

        for i in range(len(y)-self.seq_length+1):
            _x = x[i:i+self.seq_length]
            _y = y[i:i+self.seq_length]
            X_data.append(_x)
            Y_data.append(_y)
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)

        X_data, Y_data = self.suffle_array_in_the_same_order(X_data, Y_data)

        return X_data, Y_data
    def suffle_array_in_the_same_order(self,x_data,y_data):
        shuffle_index = np.arange(x_data.shape[0])
        x_data = x_data[shuffle_index]
        y_data = y_data[shuffle_index]

        return x_data, y_data
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
# file_name = 'data_diagonal.csv'
# seq_length = 10
# data_parser = DataPreprocessing(file_name,seq_length)
# X_test, Y_test = data_parser.set_test_data(isRaw=True)
#
# total_length = 0
# with open(file_name) as f:
#     for num_line, l in enumerate(f):  # For large data, enumerate should be used!
#         pass
#     total_length = num_line
# total_length +=1
# with open('results/test_diagonal_gt.csv' ,'w') as fp:
#     for i in range(int( total_length/seq_length) ):
#         np.savetxt(fp,Y_test[i],delimiter=",")
#


