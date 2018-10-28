import os
import pandas as pd

class dataProcess:
	def __init__(self, numOfMarker):
		self.cwd = os.getcwd()
		self.mkOutFolder()
		self.varGapDict = {}
		self.varList = ['X_M', 'Y_M', 'Z_M']
		self.mkVarGap(numOfMarker)

	def mkVarGap(self, numOfMarker):
		for idx in range(numOfMarker):
			for var in self.varList:
				self.varGapDict[var+str(idx + 1)] = 0

	def mkOutFolder(self):
		self.outDirPath = os.path.join(self.cwd, "changed")
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

	def doCSV(self):
		startIdx = 0
		while self.csvFile['X_M1'].iloc[startIdx] == 0:
			startIdx += 1

		print(self.csvName)
		for idx in range(startIdx, self.csvFile.shape[0]):
			if idx % 100 == 0:
				print(str(idx) + ' / ' + str(self.csvFile.shape[0]))
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
						
	def getVarGap(self, startIdx, endIdx, n):
		for var in self.varGapDict.keys():
			self.varGapDict[var] = (self.csvFile[var].iloc[endIdx] - self.csvFile[var].iloc[startIdx]) / (n + 1)

	def iterCSV(self):
		for csvName in self.csvNameList:
			self.readCSV(csvName)
			self.doCSV()
			self.writeCSV()

if __name__ == "__main__":
	dataP = dataProcess(5)
	dataP.getFolder("181025_proto_data")
	dataP.iterCSV()
