import csv
import sys
import numpy as np
import matplotlib.patches as mpatches
import cv2
import matplotlib.pyplot as plt
import numpy as np



class Calibration:
    def __init__(self):
        self.count = 0

    def set_x_y(self, x_value, y_value):
        self.x = x_value
        self.y = y_value

    def least_square(self):
        power_1 = np.power(self.x, 1)
        power_0 = np.power(self.x, 0)
        A = np.concatenate((power_1, power_0), 1)
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

    def plot_results(self, coefficient):
        self.fig = plt.figure()
        x = np.concatenate(self.x, axis = 0)
        y= np.concatenate(self.y, axis = 0)
        plt.xlim(0,5)
        plt.ylim(0,5)
        plt.scatter(x, y)

        coefficient = np.concatenate(coefficient, axis = 0)
        print (coefficient)
        least_square_x = np.linspace(0,5)
        least_square_y = coefficient[0]*np.power(least_square_x, 1)+ \
                         coefficient[1]*np.power(least_square_x, 0)
        plt.plot(least_square_x, least_square_y)

        plt.grid(True)

        self.fig = plt.gcf()
        self.fig.savefig("target" + str(self.count) +".png")
        self.count += 1

if __name__ == '__main__':
    file_name = 'extra'
    csv_1 = file_name + '_1m.csv'
    csv_2 = file_name + '_2m.csv'
    csv_3 = file_name + '_3m.csv'
    csv_4 = file_name + '_4m.csv'

    m1 = np.loadtxt(csv_1, delimiter=',')
    m2 = np.loadtxt(csv_2, delimiter=',')
    m3 = np.loadtxt(csv_3, delimiter=',')
    m4 = np.loadtxt(csv_4, delimiter=',')

    list_1 = [[1]]*len(m1)
    list_2 = [[2]]*len(m2)
    list_3 = [[3]]*len(m3)
    list_4 = [[4]]*len(m4)

    gt = np.concatenate([list_1, list_2, list_3, list_4], axis = 0)
    gt = np.array(gt)

    cali = Calibration()
    for i in range(4):
        a = []
        b = []
        c = []
        d = []
        for j in range(len(m1)):
            a.append([m1[j, 2*i]])
        for j in range(len(m2)):
            b.append([m2[j, 2*i]])
        for j in range(len(m3)):
            c.append([m3[j, 2*i]])
        for j in range(len(m4)):
            if m4[j, 2*i] == 0:
                component = (m4[j-1, 2*i] + m4[j+1, 2*i])/2
            else:
                component = m4[j, 2*i]
            d.append([component])

        y = np.concatenate([a,b,c,d], axis = 0)
        y = np.array(y)
        cali.set_x_y(y, gt)
        coefficient = cali.least_square()
        cali.plot_results(coefficient)

