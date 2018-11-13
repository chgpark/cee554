import os
import numpy as np
import pandas as pd

class dataProcess:

	def __init__(self):
		self.cwd = os.getcwd()
		self.varList = ['X_M', 'Y_M', 'Z_M']
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
		#self.csvFile['Z_M5'].apply(lambda x: x + 0.083)

		varlist = []
		for i in self.uwbSel:
			for uwb in self.uwbList:
				varlist.append(uwb + str(i))
		self.csvNP = pd.concat([self.csvFile[varlist]], axis=1).values

		varlist = ['X_M1', 'Y_M1', 'Z_M1']
		tmpNP = np.zeros((self.csvFile[varlist].values.reshape(-1,3).shape))
		for idx in range(numOfMarker):
			varlist = []
			for var in self.varList:
				varlist.append(var+str(idx+1))
			tmpNP += self.csvFile[varlist].values.reshape(-1,3)
		tmpNP /= numOfMarker
		varlist = ['X_M1', 'Y_M1', 'Z_M1']
		M1 = self.csvFile[varlist].values.reshape(-1,3) - tmpNP
		varlist = ['X_M2', 'Y_M2', 'Z_M2']
		M2 = self.csvFile[varlist].values.reshape(-1,3) - tmpNP

		for idx in range(tmpNP.shape[0]):
			tmpVec = self.normalize(np.cross(M2[idx,:].reshape(3), M1[idx,:].reshape(3)))
			tmpNP[idx, :] += tmpVec * 0.204
		self.csvNP = np.hstack((self.csvNP, tmpNP))

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

