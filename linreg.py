from sklearn import linear_model
import numpy as np
import cPickle as pickle

def loadTrainingData(userFile, quesFile, labelFile):
	userFeat = pickle.load(open(userFile, 'rb')) # user features
	quesFeat = pickle.load(open(quesFile, 'rb')) # ques features
	uLen = len(userFeat.values()[0])
	qLen = len(quesFeat.values()[0])
	# with open(userFile, 'r') as f1:
	# 	for line in f1:
	# 		sp = line.rstrip('\n').split(',')
	# 		userFeat[sp[0]] = sp[1:]
	# 		uLen = len(sp)-1
	# with open(quesFile, 'r') as f1:
	# 	for line in f1:
	# 		sp = line.rstrip('\n').split(',')
	# 		quesFeat[sp[0]] = sp[1:]
	# 		qLen = len(sp)-1

	trainingPairs = []
	with open(labelFile, 'r') as f1:
		for line in f1:
			 trainingPairs.append(line.rstrip('\r\n').split())

	x_train = np.empty(shape=(len(trainingPairs), uLen+qLen))
	y_train = np.empty(shape=(len(trainingPairs)))
	for i in range(len(trainingPairs)):
		x_train[i] = np.hstack((quesFeat[trainingPairs[i][0]],userFeat[trainingPairs[i][1]]))
		y_train[i] = trainingPairs[i][2]
	return x_train, y_train, (uLen+qLen), quesFeat, userFeat

def loadTestData(testFile, nfeat, quesFeat, userFeat):
	testData = []
	with open(testFile, 'r') as f1:
		for line in f1:
			testData.append(line.rstrip('\r\n').split(','))
	
	x_test = np.empty(shape=(len(testData[1:]), nfeat))
	for i in range(1,len(testData)):
		x_test[i-1] = np.hstack((quesFeat[testData[i][0]],userFeat[testData[i][1]]))
	
	return x_test, testData

def writeToOutputFile(x_test, y_pred, outFile, testData):
	with open(outFile, 'w') as f1:
		f1.write("qid,uid,label\n")
		y_pred[y_pred<0] = 0
		for i in range(len(x_test)):
			f1.write("{0},{1},{2}\n".format(testData[i+1][0], testData[i+1][1], y_pred[i]))





x_train, y_train, nfeat, quesFeat, userFeat = loadTrainingData('user_info_csv.dat', 'question_info_csv.dat', 'invited_info_train.txt')
print nfeat
x_test, testData = loadTestData('validate_nolabel.txt', nfeat, quesFeat, userFeat)
lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)
y_pred = model.predict(x_test)
writeToOutputFile(x_test, y_pred, 'validation_linreg.csv', testData)


