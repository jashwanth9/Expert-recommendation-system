import numpy as np 

trainData = []
vals = {}
with open('../train_data/invited_info_train.txt', 'r') as f1:
	for line in f1:
		qid, uid, val = line.rstrip('\n').split()
		trainData.append((qid, uid))
		if (qid, uid) not in vals:
			vals[(qid, uid)] = []
		vals[(qid, uid)].append(int(val))

with open('../train_data/train_norm.txt', 'w') as f1:
	for qid, uid in trainData:
		f1.write(qid+' '+uid+' '+str(sum(vals[(qid, uid)]) / float(len(vals[(qid, uid)])))+'\n')
