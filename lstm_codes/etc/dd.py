import numpy as np

a =np.loadtxt('/home/shapelim/RONet/test_karpe_1102/np_data_0.csv', delimiter =',')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(a))
# MinMaxScaler(copy=True, feature_range=(0, 1))
print(scaler.data_max_)
print(scaler.data_min_)
print(scaler.data_range_)
# print (a)
