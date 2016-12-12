# converts training data into librec format
import cPickle as pickle
import evaluate

ques_keys = pickle.load(open('../train_data/question_info_keys.dat', 'rb'))
user_keys = pickle.load(open('../train_data/user_info_keys.dat', 'rb'))
ques_keys_map = {}
user_keys_map = {}
for i in range(len(user_keys)):
	user_keys_map[user_keys[i]] = i
for i in range(len(ques_keys)):
	ques_keys_map[ques_keys[i]] = i
trainData = []
with open('../train_data/content_ques_nofeat9.txt', 'r') as f1:
	for line in f1:
		line = line.rstrip('\n')
		sp = line.split()
		trainData.append((sp[0], sp[1], float(sp[2])))

with open('../train_data/content_ques_nofeat9_librec.txt', 'w') as f1:
	for qid, uid, val in trainData:
		#if val==1:
		f1.write(str(user_keys_map[uid])+' '+str(ques_keys_map[qid])+' '+str(val)+'\n')
		#else:
			#f1.write(str(user_keys_map[uid])+' '+str(ques_keys_map[qid])+' 1\n')

valData = []

with open('../train_data/validate_nolabel.txt', 'r') as f1:
	line = f1.readline()
	for line in f1:
		valData.append(line.rstrip('\r\n').split(','))

# with open('../train_data/testdata_librec.txt', 'w') as f1:
# 	for qid, uid in valData:
# 		f1.write(str(user_keys_map[uid])+' '+str(ques_keys_map[qid])+'\n')


def convertAndEvaluate(fname, ques_keys, user_keys):
	valData = []
	outData = []
	scores = []
	with open(fname, 'r') as f1:
		header = f1.readline()
		for line in f1:
			sp = line.rstrip('\n').split()
			outData.append((ques_keys[int(sp[1])], user_keys[int(sp[0])]))
			scores.append(float(sp[3]))

	predictions = []
	maxscore = max(scores)
	minscore = min(scores)
	for i in range(len(scores)):
		predictions.append((scores[i]-minscore)/float(maxscore-minscore))

	with open(fname+'.txt', 'w') as f1:
		f1.write('qid,uid,label\n')
		for i in range(len(outData)):
			f1.write(outData[i][0] +','+outData[i][1] + ',' + str(predictions[i])+ '\n')
	print fname
	print evaluate.ndcg(fname+'.txt')


for f in ['f0', 'f3', 'f7']:
	convertAndEvaluate('../localvalidation/'+f+'.txt', ques_keys, user_keys)





# fname = '../validation/svdrow11mod.csv'
# outData = []
# scores = []
# with open('../validation/svdrow11mod_lib.txt', 'r') as f1:
# 	header = f1.readline()
# 	for line in f1:
# 		sp = line.rstrip('\n').split()
# 		outData.append((ques_keys[int(sp[1])], user_keys[int(sp[0])]))
# 		scores.append(float(sp[3]))
# predictions = {}
# maxscore = max(scores)
# minscore = min(scores)
# for i in range(len(scores)):
# 	predictions[outData[i]] = (scores[i]-minscore)/float(maxscore-minscore)

# with open(fname, 'w') as f1:
# 	f1.write('qid,uid,label\n')
# 	for i in range(len(valData)):
# 		f1.write(str(valData[i][0])+','+str(valData[i][1])+','+str(predictions[(valData[i][0], valData[i][1])])+'\n')

# print evaluate.ndcg(fname)
# print evaluate.logloss(fname)
