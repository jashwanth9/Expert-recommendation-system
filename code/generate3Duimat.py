#generates 3d spare user-item matrix U x Q X 2 (needed for latent analysis)
# and pickles it

import numpy as np
import cPickle as pickle
from scipy import sparse

ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))

useritem = np.zeros(shape=(len(user_keys), len(ques_keys), 2))
ques_keys_map = {}
user_keys_map = {}
for i in range(len(user_keys)):
	user_keys_map[user_keys[i]] = i
for i in range(len(ques_keys)):
	ques_keys_map[ques_keys[i]] = i
#positve count: 27324
#negative count: 218428
#pos label = 1
#neg label = -27324/21848 = -0.125

with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			qid, uid, val = line.split()
			if val == '1':
				useritem[user_keys_map[uid]][ques_keys_map[qid]][1] = 1
				#posc+=1
			else:
				useritem[user_keys_map[uid]][ques_keys_map[qid]][0] = 1
				#negc+=1

#print posc
#print negc
#uisparse = sparse.csr_matrix(useritem)
pickle.dump(useritem, open('../features/useritemmatrix_3d.dat', 'wb'))