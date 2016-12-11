trainData = []
with open('../train_data/train_norm.txt', 'r') as f1:
	for line in f1:
		line = line.rstrip('\n')
		sp = line.split()
		trainData.append((sp[0], sp[1], float(sp[2])))

folds = 8
i = 3
N = len(trainData)
td = trainData[:(i)*(N/folds)] + trainData[(i+1)*(N/folds):]
valData = [x[:2] for x in trainData[i*(N/folds):(i+1)*(N/folds)]]

with open('../train_data/localtraining'+str(i)+'_norm.txt', 'w') as f1:
	for qid, uid, val in td:
		f1.write(qid+' '+uid+' '+str(val)+'\n')

with open('../train_data/localvalidation'+str(i)+'.csv', 'w') as f1:
	f1.write('qid,uid,value\n')
	for qid, uid in valData:
		f1.write(qid+','+uid+'\n')


