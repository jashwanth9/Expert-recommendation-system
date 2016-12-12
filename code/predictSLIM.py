import numpy as np 
import cPickle as pickle

def getPredictions(valData, fnamepos, fnameneg, ques_keys, user_keys_map):
	slimpred = [{} for i in range(len(user_keys_map))]
	with open(fnamepos, 'r') as f1:
		i = 0
		for line in f1:
			sp = line.rstrip('\n').split()
			for j in range(len(sp)):
				if j%2 == 0:
					if int(sp[j])-1 < len(ques_keys):
						try:
							slimpred[i][ques_keys[int(sp[j])-1]] += float(sp[j+1])
						except:
							slimpred[i][ques_keys[int(sp[j])-1]] = float(sp[j+1])
					else:
						try:
							slimpred[i][ques_keys[int(sp[j])-1-len(ques_keys)]] -= (0.125)*float(sp[j+1])
						except:
							slimpred[i][ques_keys[int(sp[j])-1-len(ques_keys)]] = (-0.125)*float(sp[j+1])

			i += 1

	# with open(fnameneg, 'r') as f1:
	# 	i = 0
	# 	for line in f1:
	# 		sp = line.rstrip('\n').split()
	# 		for j in range(len(sp)):
	# 			if j%2 == 0:
	# 				try:
	# 					slimpred[i][user_keys[int(sp[j])-1]] += (-0.125)*float(sp[j+1])
	# 				except:
	# 					slimpred[i][user_keys[int(sp[j])-1]] = (-0.125)*float(sp[j+1])
	# 		i += 1

	scores = []
	for qid, uid in valData:
		if qid in slimpred[user_keys_map[uid]]:
			scores.append(slimpred[user_keys_map[uid]][qid])
		else:
			scores.append(0)
	#print scores
	predictions = []

	#normalization
	maxscore = max(scores)
	minscore = min(scores)
	for score in scores:
		predictions.append((score-minscore)/float(maxscore-minscore))
	return predictions

def loadLocalValidationData():
	testData = []
	with open('../train_data/localvalidation.txt', 'r') as f1:
		#line = f1.readline()
		for line in f1:
			testData.append(line.rstrip('\r\n').split()[:2])
	return testData



def run(valData):
	ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
	user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))
	ques_keys_map = {}
	user_keys_map = {}
	for i in range(len(user_keys)):
		user_keys_map[user_keys[i]] = i
	for i in range(len(ques_keys)):
		ques_keys_map[ques_keys[i]] = i
	predictions = getPredictions(valData, 'pred_split.txt', 'pred_neg.txt', ques_keys, user_keys_map)

	fname = '../localvalidation/slimqu_split.csv'
	with open(fname , 'w') as f1:
		f1.write('qid,uid,label\n')
		for i in range(0, len(predictions)):
			f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')
	return 
	#return evaluate.ndcg(fname)

valData = loadLocalValidationData()
run(valData)

