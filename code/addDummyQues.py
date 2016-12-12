import random

ftrain = '../train_data/localtraining7_librec_dummy.txt'
feval = '../train_data/localvalidation7_librec.txt'

quesseen = set()
with open(ftrain, 'r') as f1:
	for line in f1:
		line = line.rstrip('\n')
		sp = line.split()
		#trainData.append((sp[0], sp[1], int(sp[2])))
		quesseen.add(sp[1])

ques2 = set()
with open(feval, 'r') as f1:
	for line in f1:
		line = line.rstrip('\n')
		sp = line.split()
		#trainData.append((sp[0], sp[1], int(sp[2])))
		ques2.add(sp[1])
miss = ques2 - quesseen
print len(miss)

f1 = open(ftrain, 'a')
for q in miss:
	f1.write(str(random.choice(range(27000)))+' '+q+' 3\n')
f1.close()