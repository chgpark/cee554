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
		self.csvFile.to_csv(self.csvOutPath)
		np.savetxt(self.csvOutNpPath, self.csvFile.values, delimiter=',')

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

	def doTrainData(self):
		#self.csvFile['Z_M5'].apply(lambda x: x + 0.083)

		varlist = []
		for i in self.uwbSel:
			for uwb in self.uwbList:
				varlist.append(uwb + str(i))

		varlist.append('X_M5')
		varlist.append('Y_M5')

		self.csvFile = pd.concat([self.csvFile[varlist], self.csvFile['Z_M5'].apply(lambda x: x + 0.083)], axis=1)


			
			

	def iterProcess(self):
		for csvName in self.csvNameList:
			self.readCSV(csvName)
			self.doProcess()
			self.writeCSV()

	def iterTrainData(self):
		for csvName in self.csvNameList:
			self.readCSV(csvName)
			self.doTrainData()
			self.writeCSV()

if __name__ == "__main__":
	### data process
	#dataP = dataProcess()
	#dataP.outDirName = 'changed'
	#dataP.setProcess(5)
	#dataP.getFolder("181025_proto_data")
	#dataP.iterProcess()

	### train data
	dataT = dataProcess()
	dataT.outDirName = 'train'
	dataT.setTrainData()
	dataT.getFolder("changed")
	dataT.iterTrainData()
