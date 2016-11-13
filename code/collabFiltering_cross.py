#simplified implementation of collabarative filtering

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance 
import pdb
import warnings
from scipy import sparse
import cPickle as pickle
import evaluate

def loadData():
	#useritem_sparse = pickle.load(open('../features/useritemmatrix_normalized.dat', 'rb'))
	valData = []
	question_feats = {}
	trainData = []

	with open('../train_data/validate_nolabel.txt', 'r') as f1:
		header = f1.readline()
		for line in f1:
			valData.append(line.rstrip('\r\n').split(','))
	ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
	user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))
	
	# tf = pickle.load(open('../features/ques_charid_tfidf.dat', 'rb'))
	# tfx = tf.toarray()
	# for i in range(len(tfx)):
	# 	question_feats[question_keys[i]] = tfx[0].tolist()
	# with open('../train_data/invited_info_train.txt', 'r') as f1:
	# 	for line in f1:
	# 		line = line.rstrip('\n')
	# 		sp = line.split()
	# 		trainData.append((sp[0], sp[1], int(sp[2])))

	return ques_keys, user_keys


def collabFilteringPredictions(useritem, sparse, k, valData, ques_keys, user_keys):
	print "getting predictions"
	#input: useritem matrix
	#sparese: whether useritem is sparse or not
	#k : k nearest neighbors to consider
	#returns list of predictions
	similarities = cosine_similarity(useritem)
	scores = []
	print similarities.shape
	useritemfull = useritem.toarray()
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

def getUserItemMatrix(trainData, ques_keys, user_keys):
	print "getting useritem matrix"
	useritem = np.zeros(shape=(len(user_keys), len(ques_keys)))
	for qid, uid, val in trainData:
		if val == '1' or val==1:
			useritem[user_keys.index(uid)][ques_keys.index(qid)] = 1
				#posc+=1
		else:
			useritem[user_keys.index(uid)][ques_keys.index(qid)] = -0.125
	uisparse = sparse.csr_matrix(useritem)
	return uisparse


def run(trainData, valData, foldno, k):
	ques_keys, user_keys = loadData()
	useritem_sparse = getUserItemMatrix(trainData, ques_keys, user_keys)
	
	predictions = collabFilteringPredictions(useritem_sparse, True, k, valData, ques_keys, user_keys)

	fname = '../localvalidation/collab_norm_excludingself'+str(k)+'_'+str(foldno)+'.csv'
	with open(fname, 'w') as f1:
		f1.write('qid,uid,label\n')
		for i in range(0, len(predictions)):
			f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')

	return evaluate.ndcg(fname)




