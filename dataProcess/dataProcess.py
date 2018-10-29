import os
import pandas as pd

class dataProcess:
	def __init__(self, numOfMarker):
		self.cwd = os.getcwd()
		self.varList = ['X_M', 'Y_M', 'Z_M']
		self.outDirName = None

	def setProcess(self):
		self.mkOutFolder()
		self.varGapDict = {}
		self.mkVarGap(numOfMarker)

	def setTrainData(self):
		self.mkOutFolder()
		self.uwbList = [1, 2, 3, 4]

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
		self.csvInPath = os.path.join(self.dataFolder, self.csvName)
		self.csvFile = pd.read_csv(self.csvInPath)

	def writeCSV(self):
		self.csvOutPath = os.path.join(self.outDirPath, self.csvName)
		self.csvFile.to_csv(self.csvOutPath)

	def doProcess(self):
		startIdx = 0
		while self.csvFile['X_M1'].iloc[startIdx] == 0:
			startIdx += 1

		print(self.csvName)
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
		for idx in range(self.csvFile.shape[0]):
			
			

	def iterProcess(self):
		for csvName in self.csvNameList:
			self.readCSV(csvName)
			self.doProcess()
			self.writeCSV()

	def iterTrainData(self):
		for csvName in self.csvNameLIst:
			self.readCSV(csvName)
			self.doTrainData()
			self.writeCSV()

if __name__ == "__main__":
	### data process
	dataP = dataProcess(5, false)
	dataP.outDirName = 'changed'
	dataP.setProcess()
	dataP.getFolder("181025_proto_data")
	dataP.iterProcess()

	### train data
	dataT = dataProcess(5, true)
	dataT.outDirName = 'train'
	dataT.setTrainData()
	dataT.getFolder("changed")
	dataT.iterProcess()
