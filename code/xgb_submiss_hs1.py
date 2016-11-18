import numpy as np
import xgboost as xgb
import cPickle as pickle
import evaluate
from scipy.sparse import hstack, coo_matrix, vstack

def train_xgb(dtrain, dtest, num_round, param):
	# Training
	bst = xgb.train(param, dtrain, num_round)
	# Prediction
	ypred = bst.predict(dtest)
	# If early stopping is enabled during training, you can get predicticions from the best iteration with bst.best_ntree_limit:
	# ypred = bst.predict(xgmat,ntree_limit=bst.best_ntree_limit)
	return ypred



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

# Test data
with open('../train_data/validate_nolabel.txt') as train_file:
	content = train_file.readlines()
testData = []
element = content[1].strip("\r\n").split(",")
tdata = np.zeros(shape=(len(content)-1, question_feats[element[0]].shape[1] + user_feats[element[1]].shape[1]))
for i in range(1, len(content)):
	element = content[i].strip("\r\n").split(",")
	testData.append(element)
	tdata[i-1] = np.hstack((question_feats[element[0]].toarray(), user_feats[element[1]].toarray()))
print tdata.shape



param = {'booster':'dart', 'sample_type':'uniform' , 'objective':'binary:logistic', 'normalize_type':'forest', 'max_depth':'70', 'eta':'0.09', 'rate_drop':'0.27', 'skip_drop':'0.7'}
num_round = 630


dtrain = xgb.DMatrix(data, label=label)
dtest = xgb.DMatrix(tdata)

res = train_xgb(dtrain, dtest, num_round, param)




#normalization
# predictions = []
# scores = ypred
# maxscore = max(scores)
# minscore = min(scores)
# for score in scores:
# 	predictions.append((score-minscore)/float(maxscore-minscore))

ypred = res

with open('../validation/v_xgb_dart_forest.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for i in range(0, len(ypred)):
		f1.write(testData[i][0]+','+testData[i][1]+','+str(ypred[i])+'\n')
