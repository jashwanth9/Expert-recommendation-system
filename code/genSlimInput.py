import cPickle as pickle
import numpy as np

def genCSRFile(trainData, fname):
	ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
	user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))
	ques_keys_map = {}
	user_keys_map = {}
	for i in range(len(user_keys)):
		user_keys_map[user_keys[i]] = i
	for i in range(len(ques_keys)):
		ques_keys_map[ques_keys[i]] = i
	itemusermat = np.zeros(shape=(len(ques_keys_map), len(user_keys_map)))
	for qid, uid, val in trainData:
		if val == '1' or val==1:
			itemusermat[ques_keys_map[qid]][user_keys_map[uid]] = 1
		else:
			itemusermat[ques_keys_map[qid]][user_keys_map[uid]] = -1


	with open(fname, 'w') as f1:
		for i in range(len(user_keys_map)):
			for j in range(len(ques_keys_map)):
				if itemusermat[j][i] == 1:
					f1.write(str(j+1)+' 1 ')
				#elif itemusermat[i][j] == 1:
					#f1.write(str(j+1)+' 1 ')
			for j in range(len(ques_keys_map)):
				if itemusermat[j][i] == -1:
					f1.write(str(len(ques_keys_map)+j+1)+' 1 ')
			f1.write('\n')

def loadTrainData(fname):
	trainData = []
	with open(fname, 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			sp = line.split()
			trainData.append((sp[0], sp[1], int(sp[2])))
	return trainData


trainData = loadTrainData('../train_data/localtraining.txt')
genCSRFile(trainData, '../features/uqtrain_split.csr')
