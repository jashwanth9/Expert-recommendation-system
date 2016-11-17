import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance 
import pdb
import warnings
from scipy import sparse
import cPickle as pickle
from scipy.sparse.linalg import svds
from scipy.linalg import sqrtm
import evaluate


def loadTrainTestData():
	trainData = []
	with open('../train_data/localtraining.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
	testData = []
	with open('../train_data/localvalidation.txt', 'r') as f1:
		#line = f1.readline()
		for line in f1:
			testData.append(line.rstrip('\r\n').split(',')[:2])
	return trainData, testData

def loadData():
	# useritem_sparse = pickle.load(open('../features/useritemmatrix_normalized.dat', 'rb'))
	# valData = []
	# question_feats = {}

	# with open('../train_data/validate_nolabel.txt', 'r') as f1:
	# 	header = f1.readline()
	# 	for line in f1:
	# 		valData.append(line.rstrip('\r\n').split(','))
	ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
	user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))
	ques_keys_map = {}
	user_keys_map = {}
	for i in range(len(user_keys)):
		user_keys_map[user_keys[i]] = i
	for i in range(len(ques_keys)):
		ques_keys_map[ques_keys[i]] = i
	return ques_keys_map, user_keys_map


def getReducedMatrix(useritem_sparse, k):
	print 'svd decomposition'
	#useritem = useritem_sparse.toarray()
	u, s, v = svds(useritem_sparse, k)
	sf = np.zeros(shape=(k, k))
	for i in range(k):
		sf[i][i] = s[i]
	hsf = sqrtm(sf)
	uf = np.matmul(u, hsf)
	vf = np.matmul(hsf, v)
	vf = vf.T
	#pdb.set_trace()
	#sk = s[:k]
	#uk = u[:k, :k]
	#vk = v[:k, :k]
	return (uf, vf)

def getUserItemMatrix(trainData, ques_keys_map, user_keys_map):
	print "getting useritem matrix"
	useritem = np.zeros(shape=(len(user_keys_map), len(ques_keys_map)))
	for qid, uid, val in trainData:
		if val == '1' or val==1:
			useritem[user_keys_map[uid]][ques_keys_map[qid]] = 1.0
				#posc+=1
		else:
			useritem[user_keys_map[uid]][ques_keys_map[qid]] = -0.125
	for i in range(len(useritem[0])):
		useritem[:, i] = useritem[:, i] + np.mean(useritem[:, i])
	uisparse = sparse.csr_matrix(useritem)
	return uisparse

def getPredictions(valData, userf, itemf, ques_keys_map, user_keys_map):
	print 'getting predictions'
	scores = []
	for qid, uid in valData:
		score = np.dot(userf[user_keys_map[uid]], itemf[ques_keys_map[qid]])
		scores.append(score)
	#print scores
	predictions = []

	#normalization
	maxscore = max(scores)
	minscore = min(scores)
	for score in scores:
		predictions.append((score-minscore)/float(maxscore-minscore))

	return predictions


def run(trainData, valData, k, foldno):
	#useritem_sparse, valData, ques_keys_map, user_keys_map = loadData()
	ques_keys_map, user_keys_map = loadData()
	useritem_sparse = getUserItemMatrix(trainData, ques_keys_map, user_keys_map)
	userf, itemf = getReducedMatrix(useritem_sparse, k)
	predictions = getPredictions(valData, userf, itemf, ques_keys_map, user_keys_map)

	fname = '../localvalidation/svdnonspare_'+str(k)+'_'+str(foldno)+'.csv'
	with open(fname, 'w') as f1:
		f1.write('qid,uid,label\n')
		for i in range(0, len(predictions)):
			f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')

	return evaluate.ndcg(fname)

trainData, testData = loadTrainTestData()
print run(trainData, testData, 50, 0)