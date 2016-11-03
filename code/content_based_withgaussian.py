import numpy as np 
import pdb
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import collections
import cPickle as pickle

def loadData():
	print "loading data"
	question_keys = pickle.load(open('../features/question_info_keys.dat', 'rb'))
	question_termfreq = {}
	trainData = []
	valData = []
	question_meta = {}
	with open('../features/question_word_freq.txt', 'r') as f1:
		i = 0
		for line in f1:
			line = line.rstrip()
			wordfreq = map(int, line.split())
			question_termfreq[question_keys[i]] = wordfreq
			i += 1
	with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
	with open('../train_data/validate_nolabel.txt', 'r') as f1:
		for line in f1:
			valData.append(line.rstrip('\r\n').split(','))
	with open('../train_data/question_info.txt', 'r') as f1:
		i = 0
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			question_meta[question_keys[i]] = map(int, sp[4:7])
			i += 1
	wordvec = pickle.load(open('../features/question_word_wordvec.dat', 'rb'))
	for question in question_meta:
		question_meta[question]+=(wordvec[question_keys.index(question)].tolist())

	return question_termfreq, trainData, valData[1:], question_meta

def getModels(trainData, question_feats, question_meta):
	print "getting models"
	userX = {}
	userY = {}
	metaX = {}
	for qid, uid, val in trainData:
		if uid not in userX:
			userX[uid] = []
			userY[uid] = []
			metaX[uid] = []
		userX[uid].append(question_feats[qid])
		userY[uid].append(val)
		metaX[uid].append(question_meta[qid])
	nbmodels = {}
	gmodels = {}
	for user in userX:
		nbmodels[user] = MultinomialNB()
		nbmodels[user].fit(userX[user], userY[user])
		gmodels[user] = GaussianNB()
		gmodels[user].fit(metaX[user], userY[user])

	return nbmodels, gmodels

def getPredictions(valData, nbmodels, question_feats, gmodels, question_meta):
	print "getting predictions"
	predictions = []
	for qid, uid in valData:
		if uid not in nbmodels:
			predictions.append(0)
			continue
		prob = nbmodels[uid].predict_proba([question_feats[qid]])
		prob2 = gmodels[uid].predict_proba([question_meta[qid]])
		finalprob = 1
		if len(nbmodels[uid].classes_) == 1:
			predictions.append(0)
			continue
		if nbmodels[uid].classes_[0] == 1:
			finalprob = (prob[0][0])
		else:
			finalprob = (prob[0][1])
		if gmodels[uid].classes_[0] == 1:
			finalprob*=(prob2[0][0])
		else:
			finalprob*=(prob2[0][1])
		predictions.append(finalprob)
	return predictions

question_feats, trainData, valData, question_meta = loadData()
nbmodels, gmodels = getModels(trainData, question_feats, question_meta)
predictions = getPredictions(valData, nbmodels, question_feats, gmodels, question_meta)
with open('../validation/contentbased_word_gaussian_wordvec.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for i in range(0, len(predictions)):
		f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')

