import numpy as np 
import pdb
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import collections
import cPickle as pickle
import random
import evaluate


def loadTrainTestData():
	trainData = []
	testData = []
	with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
			testData.append([sp[0], sp[1]])
	# with open('../train_data/validate_nolabel.txt', 'r') as f1:
	# 	line = f1.readline()
	# 	for line in f1:
			# testData.append(line.rstrip('\r\n').split(','))
	return trainData, testData

def loadData():
	print "loading data"
	question_keys = pickle.load(open('../features/question_info_keys.dat', 'rb'))
	question_feats = {}
	trainData = []
	valData = []
	# with open('../features/question_word_freq.txt', 'r') as f1:
	# 	i = 0
	# 	for line in f1:
	# 		line = line.rstrip()
	# 		wordfreq = map(int, line.split())
	# 		question_feats[question_keys[i]] = wordfreq
	# 		i += 1
	#tf = pickle.load(open('../features/ques_charid_tfidf.dat', 'rb'))
	#tfx = tf.toarray()
	#print tfx.shape
	topics = []
	with open('../train_data/question_info.txt', 'r') as f1:
		for line in f1:
			topic = int(line.split()[1])
			topics.append(topic)

	# with open('../train_data/question_info.txt', 'r') as f1:
	# 	i = 0
	# 	for line in f1:
	# 		line = line.rstrip('\n')
	# 		sp = line.split()
	# 		question_feats[question_keys[i]] = map(int, sp[4:7])
	# 		i += 1
	for i in range(len(question_keys)):
		question_feats[question_keys[i]] = [1 if x == topics[i] else 0 for x in range(22)]
		# question_feats[question_keys[i]] = [1, 1, 1]

	#tf2 = pickle.load(open('../features/ques_wordid_tfidf.dat', 'rb'))
	#tfx2 = tf2.toarray()
	#for i in range(len(tfx2)):
		#question_feats[question_keys[i]].append(tfx2[])
	# with open('../train_data/invited_info_train.txt', 'r') as f1:
	# 	for line in f1:
	# 		line = line.rstrip('\n')
	# 		sp = line.split()
	# 		trainData.append((sp[0], sp[1], int(sp[2])))

	# # with open('../train_data/validate_nolabel.txt', 'r') as f1:
	# #	line = f1.readline()
	# # 	for line in f1:
	# # 		valData.append(line.rstrip('\r\n').split(','))
	# #valData = [x[:2] for x in trainData]
	# random.shuffle(trainData)

	# valData = [x[:2] for x in trainData[:int(0.15*len(trainData))]]
	# trainData = trainData[int(0.15*len(trainData)):]
	return question_feats

def getModels(trainData, question_feats):
	print "getting models"
	userX = {}
	userY = {}
	for qid, uid, val in trainData:
		if uid not in userX:
			userX[uid] = []
			userY[uid] = []
		userX[uid].append(question_feats[qid])
		userY[uid].append(val)
	nbmodels = {}
	for user in userX:
		nbmodels[user] = MultinomialNB()
		nbmodels[user].fit(userX[user], userY[user])
	#print "dumping"
	#pickle.dump(nbmodels, open('../features/usermodels_queschartfidf.dat', 'wb'))
	#exit()
	return nbmodels

def getPredictions(valData, nbmodels, question_feats):
	print "getting predictions"
	predictions = []
	for qid, uid in valData:
		if uid not in nbmodels:
			predictions.append(0)
			continue
		prob = nbmodels[uid].predict_proba([question_feats[qid]])
		if nbmodels[uid].classes_[0] == 1:
			predictions.append(prob[0][0])
		elif len(prob[0])>1:
			predictions.append(prob[0][1])
		else:
			predictions.append(0)
		#if predictions[-1] <= 0:
			#predictions[-1] = 0.111
	for i in range(len(predictions)):
		predictions[i] = (predictions[i]*4) + 1
	return predictions


def run(trainData, valData, j):

	question_feats = loadData()
	nbmodels = getModels(trainData, question_feats)
	predictions = getPredictions(testData, nbmodels, question_feats)


	fname = '../fe/content_ques_nofeat'+str(j)+'.txt'
	with open(fname , 'w') as f1:
		# f1.write('qid,uid,label\n')
		for i in range(0, len(predictions)):
			f1.write(valData[i][0]+' '+valData[i][1]+' '+str(predictions[i])+'\n')
	return 0
	# return evaluate.ndcg(fname)

if __name__ == "__main__":
	trainData, testData = loadTrainTestData()
	run(trainData, testData, 9)