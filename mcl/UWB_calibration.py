import csv
import sys
import numpy as np

gt = np.array([[0.5],
               [0.6],
               [0.7],
               [0.8],
               [0.9],
               [1.0],
               [1.1],
               [1.2],
               [1.3],
               [1.4],
               [1.5],
               [1.6],
               [1.7],
               [1.8]])

non_fog_measured_1d_LiDAR = np.array([[0.5539],
                                       [0.649],
                                       [0.757],
                                       [0.8853],
                                       [0.9603],
                                       [1.049],
                                       [1.148],
                                       [1.283],
                                       [1.376],
                                       [1.47],
                                       [1.572],
                                       [1.669],
                                       [1.75],
                                       [1.82]])

fog_measured_1d_LiDAR = np.array([[0.5646],
                                   [0.6839],
                                   [0.7703],
                                   [0.8698],
                                   [1.004],
                                   [1.089],
                                   [1.172],
                                   [1.271],
                                   [1.391],
                                   [1.48],
                                   [1.573],
                                   [1.662],
                                   [1.753],
                                   [1.821]])


class Trilateration:
    def __init__(self, x_value, y_value):
        self.x = x_value
        self.y = y_value
    def least_square(self):
        power_3 = np.power(self.x, 3)
        power_2 = np.power(self.x, 2)
        power_1 = np.power(self.x, 1)
        power_0 = np.power(self.x, 0)
        A = np.concatenate((power_3, power_2, power_1, power_0), 1)
        print (A)
        # for x_i in self.x:
        #     A_i = []
        #     while dimension + 1:
        #         x_prime = x_i**dimension
        #         dimension -= dimension
        #         A_i.append(x_prime)
        #     A.append(A_i)


        A_pseudo = np.linalg.pinv(A)
        coefficients = np.dot(A_pseudo, self.y)
        return coefficients

if __name__ == '__main__':
    trilateration = Trilateration(non_fog_measured_1d_LiDAR, gt)
    print (trilateration.least_square())

'''
fog 1d_LiDAR :        y = 0.09502746x^3 - 0.24772325x^2 + 1.179433x  - 0.10962938
non_fog_1d_LiDAR :    y = 0.15284259x^3 - 0.47716544x^2 + 1.4462534x - 0.1841911
'''