import numpy as np
import xgboost as xgb
import cPickle as pickle


## data
question_feats = {}
question_keys = pickle.load(open('../features/question_info_keys.dat', 'rb'))
ques_charid = pickle.load(open('../features/ques_charid_tfidf.dat', 'rb'))
ques_wordid = pickle.load(open('../features/ques_wordid_tfidf.dat', 'rb'))
ques_tags = pickle.load(open('../features/ques_tags.dat', 'rb'))
ques_tags = ques_tags.toarray()
ques_charid = ques_charid.toarray()
ques_wordid = ques_wordid.toarray()
for i in range(len(ques_charid)):
	question_feats[question_keys[i]] = [ques_tags[i], ques_charid[i], ques_wordid[i]]

user_feats = {}
user_keys = pickle.load(open('../features/user_info_keys.dat', 'rb'))
user_charid = pickle.load(open('../features/user_charid_tfidf.dat', 'rb'))
user_wordid = pickle.load(open('../features/user_wordid_tfidf.dat', 'rb'))
user_tags = pickle.load(open('../features/user_tags.dat', 'rb'))
user_tags = user_tags.toarray()
user_charid = user_charid.toarray()
user_wordid = user_wordid.toarray()
for i in range(len(user_charid)):
	user_feats[user_keys[i]] = [user_tags[i], user_charid[i], user_wordid[i]]



# Train data
with open('../train_data/invited_info_train.txt') as train_file:
	content = train_file.readlines()

element = content[0].strip("\n").split("\t")
no_feats = len(question_feats[element[0]][0])+len(user_feats[element[1]][0]) 
			+ len(question_feats[element[0]][1])+len(user_feats[element[1]][1])
			+ len(question_feats[element[0]][2])+len(user_feats[element[1]][2])

data = np.zeros(shape=(len(content), no_feats))
label = np.zeros(shape=(len(content),1))

for i in range(len(content)):
	element = content[i].strip("\n").split("\t")
	data[i] = np.hstack((question_feats[element[0]][0], user_feats[element[1]][0], 
						question_feats[element[0]][1], user_feats[element[1]][1],
						question_feats[element[0]][2], user_feats[element[1]][2]))
	label[i]= element[2]


# load the data
print(data.shape)
print(label.shape)
labelv = label[215501:,:]
dval = xgb.DMatrix(data[215501:,:], label=labelv)
data = data[:215500,:]
label = label[:215500,:]
dtrain = xgb.DMatrix(data, label=label)
evallist  = [(dval,'eval'), (dtrain,'train')]

##########################################################
# Test data
with open('../train_data/validate_nolabel.txt') as train_file:
	content = train_file.readlines()
testData = []
element = content[1].strip("\r\n").split(",")
no_feats = len(question_feats[element[0]][0])+len(user_feats[element[1]][0]) 
			+ len(question_feats[element[0]][1])+len(user_feats[element[1]][1])
			+ len(question_feats[element[0]][2])+len(user_feats[element[1]][2])
data = np.zeros(shape=(len(content)-1, no_feats))
for i in range(1, len(content)):
	element = content[i].strip("\r\n").split(",")
	testData.append(element)
	data[i-1] = np.hstack((question_feats[element[0]][0], user_feats[element[1]][0], 
						question_feats[element[0]][1], user_feats[element[1]][1],
						question_feats[element[0]][2], user_feats[element[1]][2]))

print data.shape
dtest = xgb.DMatrix(data)




# Booster parameters
param = {'objective':'rank:pairwise', 'max_depth':'20', 'eval_metric':'ndcg@20000', 'eta':'0.18' }

# Training
num_round = 630
bst = xgb.train(param, dtrain, num_round, evallist)

# Prediction
ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
# If early stopping is enabled during training, you can get predicticions from the best iteration with bst.best_ntree_limit:

# ypred = bst.predict(xgmat,ntree_limit=bst.best_ntree_limit)

print(len(testData))
print ypred.shape


#normalization
predictions = []
scores = ypred
maxscore = max(scores)
minscore = min(scores)
for score in scores:
	predictions.append((score-minscore)/float(maxscore-minscore))

ypred = predictions

with open('../validation/v_xgboost_word_tfidf.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for i in range(0, len(ypred)):
		f1.write(testData[i][0]+','+testData[i][1]+','+str(ypred[i])+'\n')




	
