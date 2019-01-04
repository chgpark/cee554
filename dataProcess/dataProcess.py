import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class dataProcess:

    def __init__(self):
        self.cwd = os.getcwd()
        self.varList = ['X_M', 'Z_M', 'Y_M']
        self.uwbList = ['R']#, 'RSS']
        self.outDirName = None

    def setProcess(self, numOfMarker):
        self.mkOutFolder()
        self.varGapDict = {}
        self.mkVarGap(numOfMarker)

    def setTrainData(self):
        self.mkOutFolder()
        self.uwbSel = [1, 2, 3, 4, 5, 6, 7, 8]

    def mkVarGap(self, numOfMarker):
        for idx in range(numOfMarker):
            for var in self.varList:
                self.varGapDict[var+str(idx + 1)] = 0

    def getVarGap(self, startIdx, endIdx, n):
        for var in self.varGapDict.keys():
            self.varGapDict[var] = (self.csvFile[var].iloc[endIdx] - self.csvFile[var].iloc[startIdx]) / (n + 1)

    def mkOutFolder(self):
        self.outDirPath = os.path.join(self.cwd, self.outDirName)
        try:
            os.mkdir(self.outDirPath)
        except:
            print('already exist')

    def getFolder(self, dataFolder):
        self.dataFolder = os.path.join(self.cwd, dataFolder)
        self.csvNameList = os.listdir(self.dataFolder)

    def readCSV(self, fileName):
        self.csvName = fileName
        self.csvNpName = 'np_' + fileName
        self.csvInPath = os.path.join(self.dataFolder, self.csvName)
        self.csvFile = pd.read_csv(self.csvInPath)

    def writeCSV(self):
        self.csvOutPath = os.path.join(self.outDirPath, self.csvName)
        self.csvOutNpPath = os.path.join(self.outDirPath, self.csvNpName)
        # self.csvFile.to_csv(self.csvOutPath)
        np.savetxt(self.csvOutNpPath, self.csvNP, delimiter=',')

    def doProcess(self):
        startIdx = 0
        while self.csvFile['X_M1'].iloc[startIdx] == 0:
            startIdx += 1

        for idx in range(startIdx, self.csvFile.shape[0]):
            if self.csvFile['X_M1'].iloc[idx] == 0:
                if firstFlag:
                    endIdx = idx
                    while self.csvFile['X_M1'].iloc[endIdx] == 0:
                        endIdx += 1
                        if endIdx == self.csvFile.shape[0]:
                            return
                    self.getVarGap(idx - 1, endIdx, endIdx - idx)
                    firstFlag = False
                for var in self.varGapDict.keys():
                    self.csvFile[var][idx] = self.csvFile[var].iloc[idx - 1] + self.varGapDict[var]
            else:
                firstFlag = True

    def doTrainData(self, numOfMarker):
        varlist = []
        for i in self.uwbSel:
            for uwb in self.uwbList:
                varlist.append(uwb + str(i))
        self.csvNP = pd.concat([self.csvFile[varlist]], axis=1).values

        varlist = ['X_M1', 'Z_M1', 'Y_M1']
        centerNP = np.zeros((self.csvFile[varlist].values.reshape(-1,3).shape))
        for idx in range(numOfMarker):
            varlist = []
            for var in self.varList:
                varlist.append(var+str(idx+1))
            centerNP += self.csvFile[varlist].values.reshape(-1,3)
        centerNP /= numOfMarker

        M23NP = np.zeros((self.csvFile[varlist].values.reshape(-1,3).shape))
        for idx in [0, 3]:
            varlist = []
            for var in self.varList:
                varlist.append(var+str(idx+1))
            M23NP += self.csvFile[varlist].values.reshape(-1,3)
        M23NP /= 2

        M34NP = np.zeros((self.csvFile[varlist].values.reshape(-1,3).shape))
        for idx in [0, 1]:
            varlist = []
            for var in self.varList:
                varlist.append(var+str(idx+1))
            M34NP += self.csvFile[varlist].values.reshape(-1,3)
        M34NP /= 2

        X = M23NP - centerNP
        Y = M34NP - centerNP
        Z = np.cross(X, Y)
        tmp_length = np.sqrt(X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2)
        X /= tmp_length.reshape(-1,1)

        tmp_length = np.sqrt(Y[:, 0]**2 + Y[:, 1]**2 + Y[:, 2]**2)
        Y /= tmp_length.reshape(-1,1)

        tmp_length = np.sqrt(Z[:, 0]**2 + Z[:, 1]**2 + Z[:, 2]**2)
        Z /= tmp_length.reshape(-1,1)

        alpha_list = []
        beta_list = []
        gamma_list = []
        for idx in range(centerNP.shape[0]):
            X1 = Z[idx, 1] * X[idx, 0] - Z[idx, 0] * X[idx, 1]
            X2 = Z[idx, 1] * Y[idx, 0] - Z[idx, 0] * Y[idx, 1]
            alpha = np.arctan2(X2, X1)
            if alpha == 0:
                if Z[idx, 2] > 0:
                    beta = 0
                else:
                    beta = np.pi
                gamma = -np.arctan2(X[idx, 1], X[idx, 0])
            else:
                beta = np.arctan2(np.sqrt(Z[idx, 0]**2 + Z[idx, 1]**2), Z[idx, 2])
                gamma = -np.arctan2(-Z[idx, 0], Z[idx, 1])
            tmpAngle = alpha
            alpha = -gamma
            beta = -beta
            gamma = -tmpAngle
            alpha_list.append(alpha)
            beta_list.append(beta)
            gamma_list.append(gamma)

            # R = np.array([[np.cos(alpha)*np.cos(gamma)-np.cos(beta)*np.sin(alpha)*np.sin(gamma), -np.cos(alpha)*np.sin(gamma)-np.cos(beta)*np.cos(gamma)*np.sin(alpha), np.sin(alpha)*np.sin(beta)],
            #    [np.cos(gamma)*np.sin(alpha)+np.cos(alpha)*np.cos(beta)*np.sin(gamma), np.cos(alpha)*np.cos(beta)*np.cos(gamma)-np.sin(alpha)*np.sin(gamma), -np.cos(alpha)*np.sin(beta)],
            #    [np.sin(beta)*np.sin(gamma), np.cos(gamma)*np.sin(beta), np.cos(beta)]])

        alpha_np = np.array(alpha_list)
        beta_np = np.array(beta_list)
        gamma_np = np.array(gamma_list)

        q = self.euler_quaternion(alpha_np.reshape(-1,1), beta_np.reshape(-1,1), gamma_np.reshape(-1,1))

        centerNP += Z * 0.204

        self.csvNP = np.hstack((self.csvNP, centerNP, q))
        # self.drawResult3D(centerNP[1100:2550,0], centerNP[1100:2550,1], centerNP[1100:2550,2], self.csvName)
        self.drawResult2D(centerNP[:,0], centerNP[:,1], self.csvName)

    def euler_quaternion(self, alpha, beta, gamma):
        c1 = np.cos(alpha / 2)
        c2 = np.cos(beta / 2)
        c3 = np.cos(gamma / 2)
        s1 = np.sin(alpha / 2)
        s2 = np.sin(beta / 2)
        s3 = np.sin(gamma / 2)
        x = s1 * c2 * c3 + c1 * s2 * s3
        y = c1 * s2 * c3 - s1 * c2 * s3
        z = c1 * c2 * s3 + s1 * s2 * c3
        w = c1 * c2 * c3 - s1 * s2 * s3
        return np.hstack((x,y,z,w))

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def iterProcess(self):
        for csvName in self.csvNameList:
            self.readCSV(csvName)
            self.doProcess()
            self.writeCSV()

    def iterTrainData(self, numOfMarker):
        for csvName in self.csvNameList:
            self.readCSV(csvName)
            self.doTrainData(numOfMarker)
            self.writeCSV()

    def iterData(self):
        for csvName in self.csvNameList:
            tmpNP = np.loadtxt(os.path.join(self.dataFolder, csvName), delimiter=',')
            self.drawResult3D(tmpNP[:,8],tmpNP[:,9],tmpNP[:,10],csvName)

    def drawResult3D(self, X_list, Y_list, Z_list, csvName):
        fig = plt.figure()
        fig.suptitle(csvName, fontsize=16)
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot(X_list, Y_list, Z_list)
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-4, 4)
        ax1.set_zlim(0, 2)
        ax1.set_xlabel('X axis [m]', fontsize =14)
        ax1.set_ylabel('Y axis [m]', fontsize =14)
        ax1.set_zlabel('Z axis [m]', fontsize =14)

        fig.show()

    def drawResult2D(self, X_list, Y_list, csvName):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(X_list, Y_list)
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        fig.savefig('img/'+csvName+'.png')


if __name__ == "__main__":
    ### data process
    # dataP = dataProcess()
    # dataP.outDirName = 'changed'
    # dataP.setProcess(4)
    # dataP.getFolder("181102_proto_data")
    # dataP.iterProcess()

    ### train data
    dataT = dataProcess()
    dataT.outDirName = '181102_train'
    dataT.setTrainData()
    dataT.getFolder("changed")
    dataT.iterTrainData(4)

    ### plot data
    # dataT = dataProcess()
    # dataT.getFolder("181102_train")
    # dataT.iterData()

    # X = np.array([1, 0, 0])
    # Y = np.array([0, 1, 0])
    # Z = np.array([0, 0, 1])
    #
    # X1 = Z[1] * X[0] - Z[0] * X[1]
    # X2 = Z[1] * Y[0] - Z[0] * Y[1]
    # alpha = np.arctan2(X2, X1)
    # if alpha == 0:
    #     if Z[2] > 0:
    #         beta = 0
    #     else:
    #         beta = np.pi
    #     gamma = -np.arctan2(X[1], X[0])
    # else:
    #     beta = np.arctan2(np.sqrt(Z[0]**2 + Z[1]**2), Z[2])
    #     gamma = -np.arctan2(-Z[0], Z[1])
    # print(alpha * 180 / np.pi)
    # print(beta * 180 / np.pi)
    # print(gamma * 180 / np.pi)
    #
    # R = np.array([[np.cos(alpha)*np.cos(gamma)-np.cos(beta)*np.sin(alpha)*np.sin(gamma), -np.cos(alpha)*np.sin(gamma)-np.cos(beta)*np.cos(gamma)*np.sin(alpha), np.sin(alpha)*np.sin(beta)],
    #    [np.cos(gamma)*np.sin(alpha)+np.cos(alpha)*np.cos(beta)*np.sin(gamma), np.cos(alpha)*np.cos(beta)*np.cos(gamma)-np.sin(alpha)*np.sin(gamma), -np.cos(alpha)*np.sin(beta)],
    #    [np.sin(beta)*np.sin(gamma), np.cos(gamma)*np.sin(beta), np.cos(beta)]])
    #
    # Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    # Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    # Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0],[0, 0, 1]])
    #
    # Rz1 = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0],[0, 0, 1]])
    # Rx2 = np.array([[1, 0, 0], [0, np.cos(beta), -np.sin(beta)], [0, np.sin(beta), np.cos(beta)]])
    # Rz3 = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0],[0, 0, 1]])
    #
    # Rn = np.matmul(Rz, Ry)
    # Rn = np.matmul(Rn, Rx)
    #
    # Rm = np.matmul(Rz1, Rx2)
    # Rm = np.matmul(Rm, Rz3)
    #
    # v = np.array([[1], [0], [0]])
    # v = np.matmul(R, v)
    # print(v)

