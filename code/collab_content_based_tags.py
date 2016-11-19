import pdb
import random
import evaluate
import warnings
import collections
import numpy as np 
import cPickle as pickle
from scipy import sparse
from scipy.spatial import distance 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


def loadTrainTestData():
	trainData = []
	with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
	testData = []
	with open('../train_data/validate_nolabel.txt', 'r') as f1:
		line = f1.readline()
		for line in f1:
			testData.append(line.rstrip('\r\n').split(','))
	return trainData, testData


def loadData():
	print "loading data"
	question_keys = pickle.load(open('../features/question_info_keys.dat', 'rb'))
	question_feats = {}
	trainData = []
	valData = []
	# with open('../features/question_word_freq.txt', 'r') as f1:
	# 	i = 0
	# 	for line in f1:
	# 		line = line.rstrip()
	# 		wordfreq = map(int, line.split())
	# 		question_feats[question_keys[i]] = wordfreq
	# 		i += 1
	#tf = pickle.load(open('../features/ques_charid_tfidf.dat', 'rb'))
	#tfx = tf.toarray()
	#print tfx.shape
	topics = []
	with open('../train_data/question_info.txt', 'r') as f1:
		for line in f1:
			topic = int(line.split()[1])
			topics.append(topic)

	# with open('../train_data/question_info.txt', 'r') as f1:
	# 	i = 0
	# 	for line in f1:
	# 		line = line.rstrip('\n')
	# 		sp = line.split()
	# 		question_feats[question_keys[i]] = map(int, sp[4:7])
	# 		i += 1
	for i in range(len(question_keys)):
		question_feats[question_keys[i]] = [1 if x == topics[i] else 0 for x in range(22)]
		# question_feats[question_keys[i]] = [1, 1, 1]

	#tf2 = pickle.load(open('../features/ques_wordid_tfidf.dat', 'rb'))
	#tfx2 = tf2.toarray()
	#for i in range(len(tfx2)):
		#question_feats[question_keys[i]].append(tfx2[])
	# with open('../train_data/invited_info_train.txt', 'r') as f1:
	# 	for line in f1:
	# 		line = line.rstrip('\n')
	# 		sp = line.split()
	# 		trainData.append((sp[0], sp[1], int(sp[2])))

	# # with open('../train_data/validate_nolabel.txt', 'r') as f1:
	# #	line = f1.readline()
	# # 	for line in f1:
	# # 		valData.append(line.rstrip('\r\n').split(','))
	# #valData = [x[:2] for x in trainData]
	# random.shuffle(trainData)

	# valData = [x[:2] for x in trainData[:int(0.15*len(trainData))]]
	# trainData = trainData[int(0.15*len(trainData)):]
	# useritem_sparse = pickle.load(open('../features/useritemmatrix_normalized.dat', 'rb'))
	ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
	user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))
	ques_keys_map = {}
	user_keys_map = {}
	for i in range(len(user_keys)):
		user_keys_map[user_keys[i]] = i
	for i in range(len(ques_keys)):
		ques_keys_map[ques_keys[i]] = i
	return question_feats, ques_keys_map, user_keys_map, user_keys


def getUserItemMatrix(trainData, ques_keys_map, user_keys_map):
	print "getting useritem matrix"
	useritem = np.zeros(shape=(len(user_keys_map), len(ques_keys_map)))
	for qid, uid, val in trainData:
		if val == '1' or val==1:
			useritem[user_keys_map[uid]][ques_keys_map[qid]] = 1
				#posc+=1
		else:
			useritem[user_keys_map[uid]][ques_keys_map[qid]] = -0.125
	uisparse = sparse.csr_matrix(useritem)
	return uisparse


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
	#print "dumping"
	#pickle.dump(nbmodels, open('../features/usermodels_queschartfidf.dat', 'wb'))
	#exit()
	return nbmodels


def getPredictions(valData, nbmodels, question_feats, useritem, user_keys_map, user_keys, k):
	print "getting predictions"
	similarities = cosine_similarity(useritem)
	print similarities.shape
	
	predictions = []
	i = 0
	for qid, uid in valData:
		# print i
		# i += 1
		if uid not in nbmodels:
			predictions.append(0)
			continue

		score = 0
		y = 0
		for nbindex in similarities[user_keys_map[uid]].argsort()[(-k-1):]:
			if user_keys[nbindex] not in nbmodels:
				y+=1
				sc = 0
				continue
			prob = nbmodels[user_keys[nbindex]].predict_proba([question_feats[qid]])
			if nbmodels[user_keys[nbindex]].classes_[0] == 1:
				sc = prob[0][0]
			elif len(prob[0])>1:
				sc = prob[0][1]
			else:
				y+=1
				sc = 0
			score += sc

		alt_score = score/(k-y)
		score = score/k
		# print("score:- ", score)
		# print("altscore:-", alt_score)
		prob = nbmodels[uid].predict_proba([question_feats[qid]])
		print prob
		if nbmodels[uid].classes_[0] == 1:
			predictions.append(prob[0][0]*0.75 + alt_score*0.43)
		elif len(prob[0])>1:
			predictions.append(prob[0][1]*0.75 + alt_score*0.5)
		else:
			predictions.append(alt_score*2)
		#if predictions[-1] <= 0:
			#predictions[-1] = 0.111
	print max(predictions)
	return predictions


def run(trainData, valData):
	k = 180
	question_feats, ques_keys_map, user_keys_map, user_keys = loadData()
	useritem_sparse = getUserItemMatrix(trainData, ques_keys_map, user_keys_map)
	nbmodels = getModels(trainData, question_feats)
	predictions = getPredictions(valData, nbmodels, question_feats, useritem_sparse, user_keys_map, user_keys, k)
	fname = '../validation/v_collab_alt_score.csv'
	with open(fname , 'w') as f1:
		f1.write('qid,uid,label\n')
		for i in range(0, len(predictions)):
			f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')
	#return
	return evaluate.ndcg(fname)

if __name__ == "__main__":
	trainData, testData = loadTrainTestData()
	run(trainData, testData)