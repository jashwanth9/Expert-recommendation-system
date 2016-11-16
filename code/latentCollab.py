import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance 
import pdb
from scipy import sparse
import cPickle as pickle
from scipy.sparse.linalg import svds
from scipy.linalg import sqrtm
import evaluate

#P(x|z) = alpha, P(y, v|z) = beta, P(z |x, y, v) = gamma P(z) = mixes

def loadData():
	
	valData = []
	question_feats = {}
	trainData = []
	with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))

	with open('../train_data/test_nolabel.txt', 'r') as f1:
		header = f1.readline()
		for line in f1:
			valData.append(line.rstrip('\r\n').split(','))
	ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
	user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))
	ques_keys_map = {}
	user_keys_map = {}
	for i in range(len(user_keys)):
		user_keys_map[user_keys[i]] = i
	for i in range(len(ques_keys)):
		ques_keys_map[ques_keys[i]] = i
	
	# tf = pickle.load(open('../features/ques_charid_tfidf.dat', 'rb'))
	# tfx = tf.toarray()
	# for i in range(len(tfx)):
	# 	question_feats[question_keys[i]] = tfx[0].tolist()
	# with open('../train_data/invited_info_train.txt', 'r') as f1:
	# 	for line in f1:
	# 		line = line.rstrip('\n')
	# 		sp = line.split()
	# 		trainData.append((sp[0], sp[1], int(sp[2])))

	return trainData, valData, ques_keys_map, user_keys_map


def expectation(alphas, betas, mixes, k):
	gammas = []
	totgammas = np.zeros(shape=(len(alphas), len(betas), 2))
	a = np.matmul(alphas, betas.T[0])
	b = np.matmul(alphas, betas.T[1])
	ab = np.dstack((a, b))
	for i in range(k):
		gammas.append(mixes[i]*ab)
		#print covars[i].shape
		#print means[i]
		#print data.shape
		totgammas += (mixes[i]*ab)
		#g =  multivariate_normal(mean=means[i], cov=covars[i])
		#gammas.append(mixes[i]*g.pdf(data))
		#totgammas += mixes[i]*g.pdf(data)
	for i in range(k):
		gammas[i] = gammas[i]/totgammas
	return gammas

def maximization(data, gammas, k, LEN):
	alphas = np.zeros(shape=(len(alphas), 1))
	betas = np.zeros(shape=(len(betas), 2))
	mixes = []
	N = []
	for i in range(k):
		N.append(np.sum(gammas[i]))
		mixes.append(N[i]/LEN)
		den = np.sum(data*gammas[i])
		for a in len(alphas):
			alphas[a] = np.sum(data[a]*gammas[i][a])/den
		for b in len(betas):
			for v in range(2):
				betas[b][v] = np.sum(data[:, b, v]*gammas[i][:, b, v])/den

	return alphas, betas, mixes

def getInitialEMValues(data, k):
	alphas = np.random.rand(data.shape[0], 1)
	betas = np.random.rand(data.shape[1], 2)
	mixes = [1/float(k)]*k
	return alphas, betas, mixes

def getLogLikelihood(data, alphas, betas, mixes, k):
	ll = 0
	a = np.matmul(alphas, betas.T[0].reshape(1, len(betas)))
	b = np.matmul(alphas, betas.T[1].reshape(1, len(betas)))
	ab = np.dstack((a, b))
	print ab.shape
	inner = np.zeros(shape=(len(alphas), len(betas), 2))
	for i in range(k):
		inner += mixes[i]*ab
	innerl = np.log(inner)
	ll = np.sum(innerl)
	
	return ll

def em(data, k):
	LEN = data.shape[0]*data.shape[1]*data.shape[2]
	alphas, betas, mixes = getInitialEMValues(data, k)
	ll = getLogLikelihood(data, alphas, betas, mixes, k)
	print ll
	logl = [ll]
	last = ll
	for i in range(10):
		gammas = expectation(alphas, betas, mixes, k)
		alphas, betas, mixes = maximization(data, gammas, k, LEN)
		ll = getLogLikelihood(data, alphas, betas, mixes, k)
		logl.append(ll)
		#print ll
		if ll - last < pow(10, -8):
			print i
			break
		last = ll
	print logl
	#gammas = expectation(data, means, covars, mixes, k)
	return logl, alphas, betas, mixes

def getUserItemMatrix(trainData, ques_keys_map, user_keys_map):
	print "getting useritem matrix"
	useritem = np.zeros(shape=(len(user_keys_map), len(ques_keys_map), 2))
	for qid, uid, val in trainData:
		if val == '1' or val==1:
			useritem[user_keys_map[uid]][ques_keys_map[qid]][1] = 1
				#posc+=1
		else:
			useritem[user_keys_map[uid]][ques_keys_map[qid]][0] = 1
	#uisparse = sparse.csr_matrix(useritem)
	return useritem


def run():
	trainData, valData, ques_keys_map, user_keys_map = loadData()
	useritem = getUserItemMatrix(trainData, ques_keys_map, user_keys_map)
	k = 16
	model = em(useritem, k)

run()
