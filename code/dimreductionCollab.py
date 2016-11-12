import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance 
import pdb
import warnings
from scipy import sparse
import cPickle as pickle
from numpy.linalg import svd

def loadData():
	useritem_sparse = pickle.load(open('../features/useritemmatrix_normalized.dat', 'rb'))
	valData = []
	question_feats = {}

	with open('../train_data/validate_nolabel.txt', 'r') as f1:
		header = f1.readline()
		for line in f1:
			valData.append(line.rstrip('\r\n').split(','))
	ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
	user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))

	return useritem_sparse, valData, ques_keys, user_keys


def getReducedMatrix(useritem_sparse, k):
	useritem = useritem_sparse.toarray()
	u, s, v = svd(useritem)
	sk = s[:k]
	uk = u[:k, :k]
	vk = v[:k, :k]
	pdb.set_trace()





useritem_sparse, valData, ques_keys, user_keys = loadData()
getReducedMatrix(useritem_sparse, k=14)
