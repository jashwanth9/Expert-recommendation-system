import numpy as np 
import pdb
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import collections
import cPickle as pickle
import random
import evaluate
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def loadTrainTestData():
	trainData = []
	with open('../train_data/localtraining.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
	testData = []
	with open('../train_data/localvalidation.csv', 'r') as f1:
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
		#question_feats[question_keys[i]] = [1, 1, 1]
	# with open('../features/question_char_freq.txt', 'r') as f1:
	# 	i = 0
	# 	for line in f1:
	# 		line = line.rstrip()
	# 		wordfreq = map(int, line.split())
	# 		question_feats[question_keys[i]] = wordfreq
	# 		i += 1
	user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))
	ques_keys_map = {}
	user_keys_map = {}
	for i in range(len(user_keys)):
		user_keys_map[user_keys[i]] = i
	#for i in range(len(ques_keys)):
		#ques_keys_map[ques_keys[i]] = i
	topics = []
	user_feats = []
	with open('../train_data/user_info.txt', 'r') as f1:
		for line in f1:
			topic = map(int, (line.split()[1]).split('/'))
			topics.append(topic)
	for i in range(len(topics)):
		user_feats.append([1 if x in topics[i] else 0 for x in range(145)])
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
	return question_feats, user_feats, user_keys, user_keys_map

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

def getPredictions(valData, nbmodels, question_feats, user_feats, user_keys, user_keys_map):
	print "geting similarities"
	ufsparse = sparse.csr_matrix(user_feats)
	similarities = cosine_similarity(ufsparse)
	print similarities.shape
	print "getting predictions"
	predictions = []
	k = 100
	for qid, uid in valData:
		if uid not in nbmodels:
			nscore = 0
			for nbindex in similarities[user_keys_map[uid]].argsort()[(-k-1):]:
				#nscore = 0
				cuid = user_keys[nbindex]
				if cuid in nbmodels:
					prob = nbmodels[cuid].predict_proba([question_feats[qid]])
					if nbmodels[cuid].classes_[0] == 1:
						nscore += (prob[0][0])
					elif len(prob[0])>1:
						nscore += (prob[0][1])
					else:
						nscore += (0)
			predictions.append(nscore/float(k))
			#predictions.append(0)
			continue
		prob = nbmodels[uid].predict_proba([question_feats[qid]])
		if nbmodels[uid].classes_[0] == 1:
			predictions.append(prob[0][0])
		elif len(prob[0])>1:
			predictions.append(prob[0][1])
		else:
			nscore = 0
			for nbindex in similarities[user_keys_map[uid]].argsort()[(-k-1):]:
				#nscore = 0
				cuid = user_keys[nbindex]
				if cuid in nbmodels:
					prob = nbmodels[cuid].predict_proba([question_feats[qid]])
					if nbmodels[cuid].classes_[0] == 1:
						nscore += (prob[0][0])
					elif len(prob[0])>1:
						nscore += (prob[0][1])
					else:
						nscore += (0)
			predictions.append(nscore/float(k))

			#predictions.append(0)
		#if predictions[-1] <= 0:
			#predictions[-1] = 0.111
	return predictions

def run(trainData, valData, foldno):

	question_feats, user_feats, user_keys, user_keys_map = loadData()
	nbmodels = getModels(trainData, question_feats)
	predictions = getPredictions(valData, nbmodels, question_feats, user_feats, user_keys, user_keys_map)
	fname = '../localvalidation/content_ques_topics'+str(foldno)+'.csv'
	with open(fname , 'w') as f1:
		f1.write('qid,uid,label\n')
		for i in range(0, len(predictions)):
			f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')
	return evaluate.ndcg(fname)
	print evaluate.accuracy(fname)
	print evaluate.logloss(fname)
	return
	#return evaluate.ndcg(fname)

if __name__ == "__main__":
	trainData, testData = loadTrainTestData()
	run(trainData, testData)