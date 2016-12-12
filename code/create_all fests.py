import numpy as np
import xgboost as xgb
import cPickle as pickle
from scipy.sparse import hstack, coo_matrix, vstack


## data
question_feats = {}
question_keys = pickle.load(open('../features/question_info_keys.dat', 'rb'))
ques_charid = pickle.load(open('../features/ques_charid_tfidf.dat', 'rb'))
ques_wordid = pickle.load(open('../features/ques_wordid_tfidf.dat', 'rb'))
ques_tags = pickle.load(open('../features/ques_tags.dat', 'rb'))
#ques_tags = ques_tags.toarray()
#ques_charid = ques_charid.toarray()
#ques_wordid = ques_wordid.toarray()
for i in range(len(question_keys)):
	# question_feats[question_keys[i]] = ques_charid[i]
	question_feats[question_keys[i]] = [ques_tags[i], ques_charid[i], ques_wordid[i]]

user_feats = {}
user_keys = pickle.load(open('../features/user_info_keys.dat', 'rb'))
user_charid = pickle.load(open('../features/user_charid_tfidf.dat', 'rb'))
user_wordid = pickle.load(open('../features/user_wordid_tfidf.dat', 'rb'))
user_tags = pickle.load(open('../features/user_tags.dat', 'rb'))
#user_tags = user_tags.toarray()
#user_charid = user_charid.toarray()
#user_wordid = user_wordid.toarray()
for i in range(len(user_keys)):
	# user_feats[user_keys[i]] = ser_charid[i]
	user_feats[user_keys[i]] = [user_tags[i], user_charid[i], user_wordid[i]]



# Train data
with open('../train_data/invited_info_train.txt') as train_file:
	content = train_file.readlines()

element = content[0].strip("\n").split("\t")
no_feats = (question_feats[element[0]][0].shape[1]
			+ user_feats[element[1]][0].shape[1]
			+ question_feats[element[0]][1].shape[1]
			+ user_feats[element[1]][1].shape[1]
			+ question_feats[element[0]][2].shape[1]
			+ user_feats[element[1]][2].shape[1])
print no_feats
# data = coo_matrix((len(content), no_feats))
data = hstack([question_feats[element[0]][0], user_feats[element[1]][0], 
						question_feats[element[0]][1], user_feats[element[1]][1],
						question_feats[element[0]][2], user_feats[element[1]][2]])
label = np.zeros(shape=(len(content),1))

for i in range(1,len(content)):
	print i
	element = content[i].strip("\n").split("\t")
	data_r =  hstack([question_feats[element[0]][0], user_feats[element[1]][0], 
						question_feats[element[0]][1], user_feats[element[1]][1],
						question_feats[element[0]][2], user_feats[element[1]][2]])
	data = vstack([data, data_r])
	label[i]= element[2]

inverted_all_feats = (data, label)

pickle.dump(inverted_all_feats, open('../features/allfeatures.dat', "wb"))


