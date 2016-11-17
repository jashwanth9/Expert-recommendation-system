import numpy as np
import xgboost as xgb
import cPickle as pickle
import evaluate
from scipy.sparse import hstack, coo_matrix, vstack

def train_xgb(dtrain, dtest, num_round, param, file_subname, testData):
	# Training
	bst = xgb.train(param, dtrain, num_round)
	# Prediction
	ypred = bst.predict(dtest)
	# If early stopping is enabled during training, you can get predicticions from the best iteration with bst.best_ntree_limit:
	# ypred = bst.predict(xgmat,ntree_limit=bst.best_ntree_limit)
	name = '../temp/v_xgb_tagword_'+str(file_subname)+'.csv'
	with open(name, 'w') as f1:
		f1.write('qid,uid,label\n')
        	for i in range(0, len(ypred)):
			f1.write(testData[i][0]+','+testData[i][1]+','+str(ypred[i])+'\n')

	return evaluate.ndcg(name)



## data
question_feats = {}
question_keys = pickle.load(open('../features/question_info_keys.dat', 'rb'))
tf1 = pickle.load(open('../features/ques_charid_tfidf.dat', 'rb'))
ques_tags = pickle.load(open('../features/ques_tags.dat', 'rb'))
# ques_wordid = pickle.load(open('../features/ques_wordid_tfidf.dat', 'rb'))
tf1 = tf1.toarray()
ques_tags = ques_tags.toarray()
# ques_wordid = ques_wordid.toarray()
for i in range(len(tf1)):
	question_feats[question_keys[i]] = hstack([tf1[i], ques_tags[i]])

user_feats = {}
user_keys = pickle.load(open('../features/user_info_keys.dat', 'rb'))
tf2 = pickle.load(open('../features/user_charid_tfidf.dat', 'rb'))
user_tags = pickle.load(open('../features/user_tags.dat', 'rb'))
# user_wordid = pickle.load(open('../features/user_wordid_tfidf.dat', 'rb'))
# user_wordid = user_wordid.toarray()
tf2 = tf2.toarray()
user_tags = user_tags.toarray()
for i in range(len(tf2)):
	user_feats[user_keys[i]] = hstack([tf2[i], user_tags[i]])



# Train data
with open('../train_data/invited_info_train.txt') as train_file:
	content = train_file.readlines()

N = len(content)
element = content[0].strip("\n").split("\t")
data = np.zeros(shape=(len(content), question_feats[element[0]].shape[1] + user_feats[element[1]].shape[1]))
label = np.zeros(shape=(len(content),1))
testData = []
for i in range(N):
	element = content[i].strip("\n").split("\t")
	data[i] = np.hstack((question_feats[element[0]].toarray(), user_feats[element[1]].toarray()))
	label[i]= element[2]
	testData.append([element[0],element[1]])


#mask = np.random.choice(N, N, replace=False)
#data = data[mask]
#label = label[mask]
print(data.shape)
print(label.shape)

param = {'booster':'dart', 'objective':'binary:logistic', 'max_depth':'40', 'eta':'0.18', 'silent':0 }
num_round = 100
sample_type = ["uniform", "weighted"]
normalize_type = ["forest"]
rate_drop = [0.1]
skip_drop = [0.1]

folds = 8
for st in sample_type:
	param['sample_type'] = st
	for nt in normalize_type:
		param['normalize_type'] = nt
		for rd in rate_drop:
			param['rate_drop'] = rd
			for sd in skip_drop:
				param['skip_drop'] = sd
				res = 0
				#for i in range(folds):
				i = 0
				test_data = np.vstack((data[:(i)*(N/folds)], data[(i+1)*(N/folds):]))
				test_label = np.vstack((label[:(i)*(N/folds)], label[(i+1)*(N/folds):]))
				val_data = data[i*(N/folds):(i+1)*(N/folds)]
				testData1 = testData[i*(N/folds):(i+1)*(N/folds)]

				dtrain = xgb.DMatrix(test_data, label=test_label)
				dtest = xgb.DMatrix(val_data)
				file_subname = st+nt+str(rd)+str(sd)
				res = train_xgb(dtrain, dtest, num_round, param, file_subname, testData1)
				print("Dart Booster sample_type:- "+st+"norm_type:- "+nt+"rate_drop:- "+str(rd)+"skip drop:- "+str(sd)+" Result :-" +str(res)+"\n")




##########################################################
# Test data
# with open('../train_data/validate_nolabel.txt') as train_file:
# 	content = train_file.readlines()
# testData = []
# element = content[1].strip("\r\n").split(",")
# data = np.zeros(shape=(len(content)-1, len(question_feats[element[0]])+len(user_feats[element[1]])))
# for i in range(1, len(content)):
# 	element = content[i].strip("\r\n").split(",")
# 	testData.append(element)
# 	data[i-1] = np.hstack((question_feats[element[0]], user_feats[element[1]]))
# print data.shape

# dtest = xgb.DMatrix(data)

##########################################################



#normalization
# predictions = []
# scores = ypred
# maxscore = max(scores)
# minscore = min(scores)
# for score in scores:
# 	predictions.append((score-minscore)/float(maxscore-minscore))

# ypred = predictions

# with open('../validation/v_xgboost_word_tfidf.csv', 'w') as f1:
# 	f1.write('qid,uid,label\n')
# 	for i in range(0, len(ypred)):
# 		f1.write(testData[i][0]+','+testData[i][1]+','+str(ypred[i])+'\n')
