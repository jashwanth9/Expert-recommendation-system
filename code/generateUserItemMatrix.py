#generates spare user-item matrix (1 for answered, 0 for unknown, -1 for refused to answer) 
# and pickles it

import numpy as np
import cPickle as pickle
from scipy import sparse

ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))

useritem = np.zeros(shape=(len(user_keys), len(ques_keys)))

#positve count: 27324
#negative count: 218428
#pos label = 1
#neg label = -27324/21848 = -0.125

with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			qid, uid, val = line.split()
			if val == '1':
				useritem[user_keys.index(uid)][ques_keys.index(qid)] = 1
				#posc+=1
			else:
				useritem[user_keys.index(uid)][ques_keys.index(qid)] = -0.125
				#negc+=1

#print posc
#print negc
uisparse = sparse.csr_matrix(useritem)
pickle.dump(uisparse, open('../features/useritemmatrix_normalized.dat', 'wb'))