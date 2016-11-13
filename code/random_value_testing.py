#Script to assing random uniform probabilities to question,user pairs in validate_nolabel.txt
#This can be treated as baseline
import random
import evaluate
# trainData = []
# with open('../train_data/invited_info_train.txt', 'r') as f1:
# 		for line in f1:
# 			line = line.rstrip('\n')
# 			sp = line.split()
# 			trainData.append((sp[0], sp[1], int(sp[2])))
# random.shuffle(trainData)
# valData = [x[:2] for x in trainData[:int(0.15*len(trainData))]]
# trainData = trainData[int(0.15*len(trainData)):]
# with open('../localvalidation/validation_random.csv', 'w') as f1:
# 	f1.write('qid,uid,\n')
# 	for qid, uid in valData:
# 		f1.write(qid+','+uid+','+str(random.random())+'\n')

# print evaluate.ndcg('../localvalidation/validation_random.csv')
# print evaluate.accuracy('../localvalidation/validation_random.csv')
# with open('validate_nolabel.txt', 'r') as f1:
# 	header = f1.readline()
# 	f2.write(header)
# 	for line in f1:
# 		f2.write(line.rstrip('\r\n') + ',' + str(random.random()) + '\n')
# 	f2.close()


def run(trainData, valData, foldno):
	fname = '../localvalidation/validation_random+'+str(foldno)+'.csv'
	with open(fname, 'w') as f1:
		f1.write('qid,uid,\n')
		for qid, uid in valData:
			f1.write(qid+','+uid+','+str(random.random())+'\n')

	return evaluate.ndcg(fname)
		#print evaluate.accuracy('../localvalidation/validation_random.csv')
