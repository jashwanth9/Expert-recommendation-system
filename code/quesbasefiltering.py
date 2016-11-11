import pdb
import warnings
import numpy as np
import pickle
from scipy import sparse
from scipy.spatial import distance 
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings('error')


def getItemProfiles(userData, questionData, trainData, uList, wordvec):
	# userData dict userId -> list[topics]
	# questionData dict quId -> topic (int)
	# trainData list of tuples (quId, userId, 0/1)
	# uList - list of uids (to maintain order)
	# return itemProfiles dict userId -> numpy array [features (int -2 to 2)]
	itemProfiles = {}
	len_user = len(userData)
	for ques in questionData:
		itemProfiles[ques] = np.zeros(shape=(len_user, ))
	for tup in trainData:
		qid, uid, ans = tup
		val = 1
		if ans == 0:
			val = -1*val
		itemProfiles[qid][uList.index(uid)] = val
	for item in itemProfiles.keys():
		#userProfiles[user] = np.hstack((userProfiles[user], wordvec[user]))
		if np.linalg.norm(itemProfiles[item]) == 0:
			del itemProfiles[item]
		
	return itemProfiles
	

def getKNearestUsers(userProfiles, k):
	# return k nearestUsers dict userId -> list[userIds]
	uList = userProfiles.keys()
	#samples = []
	samples = np.zeros(shape=(len(uList), len(userProfiles[uList[0]])))
	for i in range(len(uList)):
		#samples.append(userProfiles[uList[i]].tolist())
		samples[i] = userProfiles[uList[i]]
	#neigh = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=cosine_similarity)
	#pdb.set_trace()
	A_sparse = sparse.csr_matrix(samples)
	similarities = cosine_similarity(A_sparse)
	#pdb.set_trace()

	#cnt = 0
	nb = {}
	for i in range(len(uList)):
		#if np.linalg.norm(userProfiles[uList[i]]) == 0:
			#print uList[i]
			#cnt +=1
	#print cnt
	#print len(uList)sim = []
		#for j in range(len(uList)):
			#sim.append(1 - distance.cosine(userProfiles[uList[i]], userProfiles[uList[j]]))
		l1 = similarities[i].argsort()[(-k-1):]
		nb[uList[i]] = []
		for j in l1:
			if j == i:
				continue
			nb[uList[i]].append(uList[j])

	#neigh.fit(samples)
	return nb, uList


def getScores(neighbors, uList, userProfiles, valData, qList):
	scores = []
	for val in valData:
		ques, user = val
		score = 0
		if ques not in userProfiles:
			scores.append([ques, user, score])
			continue
		for nb in neighbors[ques]:
			nbprof = userProfiles[nb]
			if  nbprof[qList.index(user)] > 0:
				score += (1-distance.cosine(userProfiles[ques], nbprof))
			elif nbprof[qList.index(user)] < 0:
				score -= (1-distance.cosine(userProfiles[ques], nbprof))
		scores.append([ques, user, score])
	return scores



def loadData():
	userData = {}
	questionData = {}
	trainData = []
	valData = []
	wordvec = {}
	userorder = []
	with open('../train_data/user_info.txt', 'r') as f1:
		for line in f1:
			sp = line.split()
			uid = sp[0]
			userorder.append(uid)
			topics = sp[1].split('/')
			userData[uid] = topics
	with open('../train_data/question_info.txt', 'r') as f1:
		for line in f1:
			sp = line.split()
			qid = sp[0]
			topic = sp[1]
			questionData[qid] = topic
	with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
	with open('../train_data/validate_nolabel.txt', 'r') as f1:
		for line in f1:
			valData.append(line.rstrip('\r\n').split(','))
	#wv = pickle.load(open('user_word_wordvec.p', 'rb'))
	#for i in range(0, len(wv)):
		#wordvec[userorder[i]] = wv[i]

	return userData, questionData, trainData, valData[1:], wordvec



K=100
userData, questionData, trainData, valData, wordvec = loadData()

user_keys = userData.keys()
itemProfiles = getItemProfiles(userData, questionData, trainData, user_keys, wordvec)
nb, uList = getKNearestUsers(itemProfiles, K)

print("done1")
scores = getScores(nb, uList, itemProfiles, valData, user_keys)
umax = float("-inf")
umin = float("inf")

print("done1")
#scaling (converting to probabilities)
for score in scores:
	#if score[0] not in umax:
	umax = max(umax, score[2])
	umin = min(umin, score[2])
print("done1")
# pdb.set_trace()
with open('../validation/v_itemfilter_all'+str(K)+'.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for score in scores:
		diff = umax - umin
		if diff <= 0:
			sc = 0
		else:
			sc = (score[2]-umin)/float(diff)
		f1.write(str(score[0])+','+str(score[1])+','+str(sc)+'\n')


