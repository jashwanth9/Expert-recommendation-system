# script for collaborative filtering with K nearest users and L nearest questions
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance 
import pdb
import warnings
from scipy import sparse
import cPickle as pickle

warnings.filterwarnings('error')


def getUserProfiles(userData, questionData, trainData, qList, wordvec):
	# userData dict userId -> list[topics]
	# questionData dict quId -> topic (int)
	# trainData list of tuples (quId, userId, 0/1)
	# qList - list of qids (to maintain order)
	# return userProfiles dict userId -> numpy array [features (int -2 to 2)]
	userProfiles = {}
	for user in userData:
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
	for user in userProfiles.keys():
		if np.linalg.norm(userProfiles[user]) == 0:
			del userProfiles[user]
		

	return userProfiles

	

def getKNearestUsers(userProfiles, k):
	# return k nearestUsers dict userId -> list[userIds]
	uList = userProfiles.keys()
	samples = np.zeros(shape=(len(uList), len(userProfiles[uList[0]])))
	for i in range(len(uList)):
		samples[i] = userProfiles[uList[i]]
	A_sparse = sparse.csr_matrix(samples)
	similarities = cosine_similarity(A_sparse)

	nb = {}
	for i in range(len(uList)):
		l1 = similarities[i].argsort()[(-k-1):]
		nb[uList[i]] = []
		for j in l1:
			if j == i:
				continue
			nb[uList[i]].append(uList[j])
	return nb, uList

def getLNearestQues(quesSim, L, qList):
	nb = {}
	for i in range(len(qList)):
		l1 = quesSim[i].argsort()[(-L):]
		nb[qList[i]] = []
		for j in l1:
			nb[qList[i]].append(qList[j])
	return nb



def getScores(neighbors, uList, userProfiles, valData, qList, qnb, quesSim, questionData, qtopic):
	scores = []
	for val in valData:
		ques, user = val
		score = 0
		if user not in userProfiles:
			scores.append([ques, user, score])
			continue
		for nb in neighbors[user]:
			nbprof = userProfiles[nb]	
			for q in qtopic[questionData[ques]]:
				if  nbprof[q] > 0:
					score += (1-distance.cosine(userProfiles[user], nbprof))
					break
				elif nbprof[q] < 0:
					score -= (1-distance.cosine(userProfiles[user], nbprof))
					break

			# for q in qnb[ques]:

			# 	if  nbprof[qList.index(q)] > 0:
			# 		score += quesSim[qList.index(ques), qList.index(q)]*(1-distance.cosine(userProfiles[user], nbprof))
			# 	elif nbprof[qList.index(q)] < 0:
			# 		score -= quesSim[qList.index(ques), qList.index(q)]*(1-distance.cosine(userProfiles[user], nbprof))
		scores.append([ques, user, score])
	return scores




def loadData():
	userData = {}
	questionData = {}
	trainData = []
	valData = []
	wordvec = {}
	userorder = []
	qtopic = {}

	with open('user_info.txt', 'r') as f1:
		for line in f1:
			sp = line.split()
			uid = sp[0]
			userorder.append(uid)
			topics = sp[1].split('/')
			userData[uid] = topics
	with open('question_info.txt', 'r') as f1:
		i = 0
		for line in f1:
			sp = line.split()
			qid = sp[0]
			topic = sp[1]
			questionData[qid] = topic
			if topic not in qtopic:
				qtopic[topic] = set()
			qtopic[topic].add(i)
			i+=1
	with open('invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
	with open('validate_nolabel.txt', 'r') as f1:
		for line in f1:
			valData.append(line.rstrip('\r\n').split(','))
	quesSim = []
	#quesSim = pickle.load(open('question_word_wordvec_sim.p', 'rb'))
	print "Data Loaded"
	#wv = pickle.load(open('user_word_wordvec.p', 'rb'))
	#for i in range(0, len(wv)):
		#wordvec[userorder[i]] = wv[i]


	#pdb.set_trace()	
	return userData, questionData, trainData, valData[1:], wordvec, quesSim, qtopic

K = 100
L = 5
userData, questionData, trainData, valData, wordvec, quesSim, qtopic = loadData()
qList = pickle.load(open('question_info_keys.dat'))
userProfiles = getUserProfiles(userData, questionData, trainData, qList, wordvec)
unb, uList = getKNearestUsers(userProfiles, K)
qnb = {}
#qnb = getLNearestQues(quesSim, L, qList)
#pdb.set_trace()

scores = getScores(unb, uList, userProfiles, valData, qList, qnb, quesSim, questionData, qtopic)
umax = float("-inf")
umin = float("inf")


#scaling (converting to probabilities)
for score in scores:
	#if score[0] not in umax:
	umax = max(umax, score[2])
	umin = min(umin, score[2])

#pdb.set_trace()
with open('v_userfilter_ques_'+str(K)+'_'+str(L)+'.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for score in scores:
		diff = umax - umin
		if diff <= 0:
			sc = 0
		else:
			sc = (score[2]-umin)/float(diff)
		f1.write(str(score[0])+','+str(score[1])+','+str(sc)+'\n')


