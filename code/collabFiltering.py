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
	useritem_sparse = pickle.load(open('../features/useritemmatrix.dat', 'rb'))
	valData = []
	with open('../train_data/validate_nolabel.txt', 'r') as f1:
		header = f1.readline()
		for line in f1:
			valData.append(line.rstrip('\r\n').split(','))
	ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
	user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))

	return useritem_sparse, valData, ques_keys, user_keys


def collabFilteringPredictions(useritem, sparse, k, valData, ques_keys, user_keys):
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
			score += useritemfull[nbindex][ques_keys.index(qid)]*similarities[user_keys.index(uid)][nbindex]
		scores.append(score)

	predictions = []
	maxscore = max(scores)
	minscore = min(scores)
	for score in scores:
		predictions.append((score-minscore)/float(maxscore-minscore))

	return predictions

k = 20
useritem_sparse, valData, ques_keys, user_keys = loadData()
predictions = collabFilteringPredictions(useritem_sparse, True, k, valData, ques_keys, user_keys)

with open('../validation/collab_'+str(k)+'.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for i in range(0, len(predictions)):
		f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')




