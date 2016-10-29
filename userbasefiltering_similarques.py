import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance 
import pdb
import warnings
from scipy import sparse
import cPickle as pickle

#extension of user based filtering where we also look at user who have answered L similar question



def getUserProfiles(userData, questionData, trainData, qList, wordvec):
	# userData dict userId -> list[topics]
	# questionData dict quId -> topic (int)
	# trainData list of tuples (quId, userId, 0/1)
	# qList - list of qids (to maintain order)
	# return userProfiles dict userId -> numpy array [features (int -2 to 2)]
	userProfiles = {}
	quesAnswered = {}
	for user in userData:
		#userProfiles[user] = np.zeros(shape=(len(questionData), ))
		userProfiles[user] = np.zeros(shape=(len(questionData), ))
	for tup in trainData:
		qid, uid, ans = tup
		if questionData[qid] in userData[uid]:
			val = 1
		else:
			val = 1
		if ans == 0:
			val = -1*val
		userProfiles[uid][qList.index(qid)] = val
		if ans == 1:
			try:
				quesAnswered[uid].add(qid)
			except:
				quesAnswered[uid] = set()
				quesAnswered[uid].add(qid)
	for user in userProfiles.keys():
		userProfiles[user] = np.hstack((userProfiles[user], wordvec[user]))
		if np.linalg.norm(userProfiles[user]) == 0:
			del userProfiles[user]
		

	return userProfiles, quesAnswered

def compute_l2_vec(train_feats, test_feats):
	#sqrt((x-y)**2) = sqrt(x**2 + y**2 - 2x*y)
	print "here4"
	sq_sum_train = np.sum(np.square(train_feats), axis=1)
	sq_sum_test = np.sum(np.square(test_feats), axis=1)
	inner_product = np.dot(test_feats, train_feats.T)
	return np.sqrt(sq_sum_train - 2 * inner_product + sq_sum_test.reshape(-1,1))

	

def getKNearestEntities(entityProfiles, k, isSparse):
	print "here"
	# return k nearestUsers dict userId -> list[userIds]
	eList = entityProfiles.keys()
	#samples = []
	samples = np.zeros(shape=(len(eList), len(entityProfiles[eList[0]])))
	for i in range(len(eList)):
		#samples.append(userProfiles[uList[i]].tolist())
		samples[i] = entityProfiles[eList[i]]
	#neigh = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=cosine_similarity)
	#pdb.set_trace()
	print "here2"
	if isSparse:
		A_sparse = sparse.csr_matrix(samples)
		similarities = cosine_similarity(A_sparse)
	else:
		neigh = NearestNeighbors(n_neighbors=k+1)
		neigh.fit(samples) 
		x = neigh.kneighbors(samples)
		#pdb.set_trace()

		#similarities = -1*compute_l2_vec(samples, samples)
	print "here3"
	#pdb.set_trace()


	#cnt = 0
	nb = {}
	for i in range(len(eList)):
		if not isSparse:
			nb[eList[i]] = x[1][i].tolist()[1:]
			continue
		#if np.linalg.norm(userProfiles[uList[i]]) == 0:
			#print uList[i]
			#cnt +=1
	#print cnt
	#print len(uList)sim = []
		#for j in range(len(uList)):
			#sim.append(1 - distance.cosine(userProfiles[uList[i]], userProfiles[uList[j]]))
		l1 = similarities[i].argsort()[(-k-1):]
		nb[eList[i]] = []
		for j in l1:
			if j == i:
				continue
			nb[eList[i]].append(eList[j])

	#neigh.fit(samples)
	return nb, eList
def getScores(neighbors, uList, userProfiles, valData, qList, quesAnswered):
	scores = []
	for val in valData:
		ques, user = val
		score = 0
		if user not in userProfiles:
			scores.append([ques, user, score])
			continue
		for nb in neighbors[user]:
			#nbprof = userProfiles[nb]
			#if  nbprof[qList.index(ques)] > 0:
			if uList[nb] in quesAnswered and ques in quesAnswered[uList[nb]]:
				score += (1-distance.cosine(userProfiles[user], userProfiles[uList[nb]]))
		scores.append([ques, user, score])
	return scores



def loadData():
	userData = {}
	questionData = {}
	trainData = []
	valData = []
	user_wordvec = {}
	userorder = []
	ques_wordvec = {}
	with open('user_info.txt', 'r') as f1:
		for line in f1:
			sp = line.split()
			uid = sp[0]
			userorder.append(uid)
			topics = sp[1].split('/')
			userData[uid] = topics
	with open('question_info.txt', 'r') as f1:
		for line in f1:
			sp = line.split()
			qid = sp[0]
			topic = sp[1]
			questionData[qid] = topic
	with open('invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
	with open('validate_nolabel.txt', 'r') as f1:
		for line in f1:
			valData.append(line.rstrip('\r\n').split(','))
	wv = pickle.load(open('user_word_wordvec.p', 'rb'))
	for i in range(0, len(wv)):
		user_wordvec[userorder[i]] = wv[i]



	return userData, questionData, trainData, valData[1:], user_wordvec, ques_wordvec

K=30
L=10
userData, questionData, trainData, valData, user_wordvec, ques_wordvec = loadData()
qList = questionData.keys()
userProfiles, quesAnswered = getUserProfiles(userData, questionData, trainData, qList, user_wordvec)
#pdb.set_trace()
nb, uList = getKNearestEntities(userProfiles, K, False)
#getLNearestQuestions(questionData, qList, ques_wordvec)
scores = getScores(nb, uList, userProfiles, valData, qList, quesAnswered)
umax = {}
umin = {}

#scaling (converting to probabilities)
for score in scores:
	if score[1] not in umax:
		umax[score[1]] = 0
		umin[score[1]] = 0
	umax[score[1]] = max(umax[score[1]], score[2])
	umin[score[1]] = min(umin[score[1]], score[2])

with open('v_userfilter_onlywordvec'+str(K)+'.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for score in scores:
		diff = umax[score[1]] - umin[score[1]]
		if diff == 0:
			sc = 0
		else:
			sc = (score[2]-umin[score[1]])/float(diff)
		f1.write(str(score[0])+','+str(score[1])+','+str(sc)+'\n')


