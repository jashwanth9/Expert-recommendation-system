#generates spare user-item matrix (1 for answered, 0 for unknown, -1 for refused to answer) 
# and pickles it

import numpy as np
import cPickle as pickle
from scipy import sparse

ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))

useritem = np.zeros(shape=(len(user_keys), len(ques_keys)))

with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			line = line.rstrip('\n')
			qid, uid, val = line.split()
			if val == '1':
				useritem[user_keys.index(uid)][ques_keys.index(qid)] = 1
			else:
				useritem[user_keys.index(uid)][ques_keys.index(qid)] = -1


uisparse = sparse.csr_matrix(useritem)
pickle.dump(uisparse, open('../features/useritemmatrix.dat', 'wb'))