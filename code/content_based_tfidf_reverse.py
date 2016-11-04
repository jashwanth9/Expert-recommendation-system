import numpy as np 
import pdb
from sklearn.naive_bayes import MultinomialNB
import collections
import cPickle as pickle

def loadData():
	print "loading data"
	user_keys = pickle.load(open('../features/user_info_keys.dat', 'rb'))
	user_feats = {}
	trainData = []
	valData = []
	# with open('../features/question_word_freq.txt', 'r') as f1:
	# 	i = 0
	# 	for line in f1:
	# 		line = line.rstrip()
	# 		wordfreq = map(int, line.split())
	# 		question_feats[question_keys[i]] = wordfreq
	# 		i += 1
	tf = pickle.load(open('../features/user_wordid_tfidf.dat', 'rb'))
	tfx = tf.toarray()
	for i in range(len(tfx)):
		user_feats[user_keys[i]] = tfx[0].tolist()
	with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
	with open('../train_data/validate_nolabel.txt', 'r') as f1:
		for line in f1:
			valData.append(line.rstrip('\r\n').split(','))

	return user_feats, trainData, valData[1:]

def getModels(trainData, user_feats):
	print "getting models"
	quesX = {}
	quesY = {}
	for qid, uid, val in trainData:
		if qid not in quesX:
			quesX[qid] = []
			quesY[qid] = []
		quesX[qid].append(user_feats[uid])
		quesY[qid].append(val)
	nbmodels = {}
	for ques in quesX:
		nbmodels[ques] = MultinomialNB()
		nbmodels[ques].fit(quesX[ques], quesY[ques])

	return nbmodels

def getPredictions(valData, nbmodels, user_feats):
	print "getting predictions"
	predictions = []
	for qid, uid in valData:
		if qid not in nbmodels:
			predictions.append(0)
			continue
		prob = nbmodels[qid].predict_proba([user_feats[uid]])
		if nbmodels[qid].classes_[0] == 1:
			predictions.append(prob[0][0])
		else:
			predictions.append(prob[0][1])
	return predictions

user_feats, trainData, valData = loadData()
nbmodels = getModels(trainData, user_feats)
predictions = getPredictions(valData, nbmodels, user_feats)
with open('../validation/content_word_tfidf_rev.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for i in range(0, len(predictions)):
		f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')

