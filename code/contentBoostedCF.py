#simplified implementation of collabarative filtering

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance 
import pdb
import warnings
from scipy import sparse
import cPickle as pickle

def loadData():
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
	
	tf = pickle.load(open('../features/ques_charid_tfidf.dat', 'rb'))
	tfx = tf.toarray()
	for i in range(len(tfx)):
		question_feats[question_keys[i]] = tfx[0].tolist()
	with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))

	return useritem_sparse, valData, ques_keys, user_keys, trainData, question_feats


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
	for i in range(0, len(user_keys)):
		for j in range(0, len(ques_keys)):
			if useritem[i][j] == 0:
				prediction = usermodels[i].pred([question_feats[ques_keys[j]]])[0]
				if prediction == 1:
					useritem[i][j] = 1
				elif prediction == -1:
					useritem[i][j] = -0.125
				else:
					print prediction
	return useritem





def collabFilteringPredictions(useritem, sparse, k, valData, ques_keys, user_keys):
	print "collab filtering"
	#input: useritem matrix
	#sparese: whether useritem is sparse or not
	#k : k nearest neighbors to consider
	#returns list of predictions
	similarities = cosine_similarity(useritem)
	scores = []
	print similarities.shape
	if sparse:
		useritemfull = useritem.toarray()
	else:
		useritemfull = useritem
	for qid, uid in valData:
		score = 0
		for nbindex in similarities[user_keys.index(uid)].argsort()[(-k-1):]:
			if nbindex == user_keys.index(uid): #exclude self
				continue
			score += useritemfull[nbindex][ques_keys.index(qid)]*similarities[user_keys.index(uid)][nbindex]
		scores.append(score)

	predictions = []

	#normalization
	maxscore = max(scores)
	minscore = min(scores)
	for score in scores:
		predictions.append((score-minscore)/float(maxscore-minscore))

	return predictions

k = 20

useritem_sparse, valData, ques_keys, user_keys, train_data, question_feats = loadData()
usermodels = getModels(trainData, question_feats)
useritem = contentBoosting(user_keys, ques_keys, useritem_spare, usermodels, question_feats)
predictions = collabFilteringPredictions(useritem, False, k, valData, ques_keys, user_keys)

with open('../validation/content_boosted_'+str(k)+'.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for i in range(0, len(predictions)):
		f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')




