import numpy as np 
import cPickle as pickle

def getPredictions(valData, fname, ques_keys_map, user_keys):
	slimpred = [{} for i in range(len(ques_keys_map))]
	with open(fname, 'r') as f1:
		i = 0
		for line in f1:
			sp = line.rstrip('\n').split()
			for j in range(len(sp)):
				if j%2 == 0:
					slimpred[i][user_keys[int(sp[j])-1]] = float(sp[j+1])
			i += 1
			
	predictions = []
	for qid, uid in valData:
		if uid in slimpred[ques_keys_map[qid]]:
			predictions.append(slimpred[ques_keys_map[qid]][uid])
		else:
			predictions.append(0)

	return predictions

def loadValidationData():
	testData = []
	with open('../train_data/validate_nolabel.txt', 'r') as f1:
		line = f1.readline()
		for line in f1:
			testData.append(line.rstrip('\r\n').split(','))
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
	predictions = getPredictions(valData, 'pred.txt', ques_keys_map, user_keys)

	fname = '../validation/slimqu.csv'
	with open(fname , 'w') as f1:
		f1.write('qid,uid,label\n')
		for i in range(0, len(predictions)):
			f1.write(valData[i][0]+','+valData[i][1]+','+str(predictions[i])+'\n')
	return 
	#return evaluate.ndcg(fname)

valData = loadValidationData()
run(valData)

