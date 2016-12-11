from operator import itemgetter
from ndcg import ndcg_at_k
import numpy as np
from sklearn import metrics
def getTrueVal():
	trueVal = {}
	with open('../train_data/invited_info_train.txt', 'r') as f1:
		for line in f1:
			sp = line.rstrip('\n').split()
			trueVal[(sp[0], sp[1])] = int(sp[2])

	return trueVal

def ndcg(valfile):
	trueVal = getTrueVal()
	predProb = {}
	with open(valfile, 'r') as f1:
		line = f1.readline()
		for line in f1:
			qid, uid, prob = line.rstrip('\n').split(',')
			if qid not in predProb:
				predProb[qid] = []
			predProb[qid].append((uid, float(prob)))
	scores = []
	weights = []
	for qid in predProb:
		ranks = sorted(predProb[qid],key=itemgetter(1),reverse=True)
		r = []
		for rank in ranks:
			r.append(trueVal[(qid, rank[0])])
		s5 = ndcg_at_k(r, 5)
		s10 = ndcg_at_k(r, 10)
		scores.append(s5*0.5 + s10*0.5)
		weights.append(len(r))
	#print scores
	#print weights
	return np.average(scores, weights=weights)


def accuracy(valfile):
	trueVal = getTrueVal()
	cor = 0
	tot = 0
	with open(valfile, 'r') as f1:
		line = f1.readline()
		for line in f1:
			qid, uid, prob = line.rstrip('\n').split(',')
			if float(prob) > 0.5:
				pred = 1
			else:
				pred = 0

			if trueVal[(qid, uid)] == pred:
				cor += 1
			tot += 1
	return cor/float(tot)

def logloss(valfile):
	trueVal = getTrueVal()
	tv = []
	pv = []
	with open(valfile, 'r') as f1:
		line = f1.readline()
		for line in f1:
			qid, uid, prob = line.rstrip('\n').split(',')
			tv.append(trueVal[(qid, uid)])
			pv.append(float(prob))

	return metrics.log_loss(tv, pv)

