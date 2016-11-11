import numpy as np
import xgboost as xgb
import cPickle as pickle


## data
question_feats = {}
question_keys = pickle.load(open('../features/question_info_keys.dat', 'rb'))
tf1 = pickle.load(open('../features/ques_charid_tfidf.dat', 'rb'))
tf1_x = tf1.toarray()
for i in range(len(tf1_x)):
	question_feats[question_keys[i]] = tf1_x[i]

user_feats = {}
user_keys = pickle.load(open('../features/user_info_keys.dat', 'rb'))
tf2 = pickle.load(open('../features/user_charid_tfidf.dat', 'rb'))
tf2_x = tf2.toarray()
for i in range(len(tf2_x)):
	user_feats[user_keys[i]] = tf2_x[i]

# Train data
with open('../train_data/invited_info_train.txt') as train_file:
	content = train_file.readlines()

element = content[0].strip("\n").split("\t")
data = np.zeros(shape=(len(content), len(question_feats[element[0]])+len(user_feats[element[1]])))
label = np.zeros(shape=(len(content),1))

for i in range(len(content)):
	element = content[i].strip("\n").split("\t")
	data[i] = np.hstack((question_feats[element[0]], user_feats[element[1]]))
	label[i]= element[2]


# load the data
label = np.array(label).transpose()
print(data.shape)
print(label.shape)
dtrain = xgb.DMatrix(data, label=label)


##########################################################
# Test data
with open('../train_data/validate_nolabel.txt') as train_file:
	content = train_file.readlines()
testData = []
element = content[0].strip("\r\n").split(",")
data = np.zeros(shape=(len(content), len(question_feats[element[0]])+len(user_feats[element[1]])))
for i in range(len(content)):
	element = content[i].strip("\r\n").split(",")
	testData.append(element)
	data[i] = np.hstack((question_feats[element[0]], user_feats[element[1]]))


dtest = xgb.DMatrix(data)

##########################################################

# To load a scpiy.sparse array into DMatrix, the command is:
# csr = scipy.sparse.csr_matrix((dat, (row, col)))
# dtrain = xgb.DMatrix(csr)


# Booster parameters
param = {'objective':'binary:logistic' }
# param['nthread'] = 4
# param['eval_metric'] = 'auc'
# You can also specify multiple eval metrics:
# param['eval_metric'] = ['auc', 'ams@0'] 

# # alternativly:
# # plst = param.items()
# # plst += [('eval_metric', 'ams@0')]
# Specify validations set to watch performance
# evallist  = [(dtest,'eval'), (dtrain,'train')]



# With parameter list and data, you are able to train a model.

# Training
num_round = 10
bst = xgb.train(param, dtrain, num_round)

# Prediction
# # 7 entities, each contains 10 features
# data = np.random.rand(7, 10)
# dtest = xgb.DMatrix(data)

ypred = bst.predict(xgmat)
# If early stopping is enabled during training, you can get predicticions from the best iteration with bst.best_ntree_limit:

# ypred = bst.predict(xgmat,ntree_limit=bst.best_ntree_limit)

with open('../validation/v_xgboost_word_tfidf.csv', 'w') as f1:
	f1.write('qid,uid,label\n')
	for i in range(0, len(ypred)):
		f1.write(testData[i][0]+','+testData[i][1]+','+str(ypred[i])+'\n')




	