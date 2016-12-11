#simplified implementation of collabarative filtering

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance 
import pdb
import warnings
from scipy import sparse
import cPickle as pickle
from sklearn.naive_bayes import MultinomialNB
import collections


def loadTrainTestData():
	trainData = []
	with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
	testData = []
	with open('../train_data/test_nolabel.txt', 'r') as f1:
		line = f1.readline()
		for line in f1:
			testData.append(line.rstrip('\r\n').split(','))
	return trainData, testData

def loadData():
	print "loading data"
	useritem_sparse = pickle.load(open('../features/useritemmatrix_normalized.dat', 'rb'))
	valData = []
	question_feats = {}
	trainData = []

	with open('../train_data/validate_nolabel.txt', 'r') as f1:
		header = f1.readline()
		for line in f1:
			valData.append(line.rstrip('\r\n').split(','))
	ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
	user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))
	user_keys_map = {}
	ques_keys_map = {}
	for i in range(len(user_keys)):
		user_keys_map[user_keys[i]] = i
	for i in range(len(ques_keys)):
		ques_keys_map[ques_keys[i]] = i
	
	# tf = pickle.load(open('../features/ques_charid_tfidf.dat', 'rb'))
	# tfx = tf.toarray()
	# for i in range(len(tfx)):
	# 	question_feats[ques_keys[i]] = tfx[0].tolist()
	topics = []
	with open('../train_data/question_info.txt', 'r') as f1:
		for line in f1:
			topic = int(line.split()[1])
			topics.append(topic)
	for i in range(len(ques_keys)):
		question_feats[ques_keys[i]] = [1 if x == topics[i] else 0 for x in range(22)]
	with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))

	return useritem_sparse, valData, ques_keys, user_keys, trainData, question_feats, ques_keys_map, user_keys_map


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
	
	return nbmodels



def contentBoosting(user_keys, ques_keys, useritem, usermodels, question_feats):
	print "boosting"
	useritem = useritem.toarray()
	topredict = [question_feats[ques_keys[i]] for i in range(len(ques_keys))]
	for i in range(0, len(user_keys)):
		if user_keys[i] not in usermodels:
			continue
		predictions = usermodels[user_keys[i]].predict(topredict)
		for j in range(0, len(ques_keys)):
			if useritem[i][j] == 0:
				prediction = predictions[j]
				if prediction == 1:
					useritem[i][j] = 1
				elif prediction == 0:
					useritem[i][j] = -0.125
				else:
					print prediction
	return useritem





def collabFilteringPredictions(useritem, sparse, k, valData, ques_keys_map, user_keys_map):
	print "getting predictions"
	#input: useritem matrix
	#sparese: whether useritem is sparse or not
	#k : k nearest neighbors to consider
	#returns list of predictions
	similarities = cosine_similarity(useritem)
	scores = []
	print similarities.shape
	useritemfull = useritem
	for qid, uid in valData:
		score = 0
		for nbindex in similarities[user_keys_map[uid]].argsort()[(-k-1):]:
			if nbindex == user_keys_map[uid]: #exclude self
				continue
			score += useritemfull[nbindex][ques_keys_map[qid]]*similarities[user_keys_map[uid]][nbindex]
		scores.append(score)

	predictions = []

	#normalization
	maxscore = max(scores)
	minscore = min(scores)
	for score in scores:
		predictions.append((score-minscore)/float(maxscore-minscore))

	return predictions

k = 180

useritem_sparse, valData, ques_keys, user_keys, trainData, question_feats, ques_keys_map, user_keys_map = loadData()
usermodels = getModels(trainData, question_feats)
useritem = contentBoosting(user_keys, ques_keys, useritem_sparse, usermodels, question_feats)
predictions = collabFilteringPredictions(useritem, False, k, valData, ques_keys_map, user_keys_map)

with open('../validation/content_boosted_'+str(k)+'.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for i in range(0, len(predictions)):
		f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')




