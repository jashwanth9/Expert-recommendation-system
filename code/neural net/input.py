import cPickle
import numpy as np

f = open('bytecup2016data/invited_info_train.txt', 'r')
ground_truth = f.readlines()
f.close()
f = open('bytecup2016data/user_info.txt', 'r')
users = f.readlines()
f.close()
f = open('bytecup2016data/question_info.txt', 'r')
ques = f.readlines()
f.close()
f = open('bytecup2016data/validate_nolabel.txt', 'r')
_ = f.readline()
val = f.readlines()
f.close()
f = open('bytecup2016data/test_data.txt')
_ = f.readline()
test = f.readlines()
f.close()
max_topic_id = 143

user_wordvecs = cPickle.load(open('user_word_wordvec100.p', 'rb'))
user_charvecs = cPickle.load(open('user_char_wordvec100.p', 'rb'))
ques_charvecs = cPickle.load(open('question_char_wordvec100.p', 'rb'))
ques_wordvecs = cPickle.load(open('question_word_wordvec100.p', 'rb'))

user_info = {}
ques_info = {}

i = 0
for user in users:
	info = user.split('\t')
	ufeats = np.zeros(max_topic_id)
	tags = map(int, info[1].split('/'))
	ufeats[tags] = 1
	ufeats = np.hstack((ufeats, user_wordvecs[i], user_charvecs[i]))
	user_info[info[0]] = np.array(ufeats)
	i += 1

i = 0
for q in ques:
	q = q.replace('\n', '').replace('\r', '')
	info = q.split('\t')
	qfeats = []
	qfeats.append(int(info[1]))
	qfeats.append(int(info[4]))
	qfeats.append(int(info[5]))
	qfeats.append(int(info[6]))
	qfeats = np.array(qfeats)
	qfeats = np.hstack((qfeats, ques_wordvecs[i], ques_charvecs[i]))
	ques_info[info[0]] = qfeats
	i += 1
total_records = len(ground_truth)

for i in range(len(ground_truth)):
	record = ground_truth[i]
	record = record.replace('\n', '').replace('\r', '')
	record = record.split('\t')
	q_id = record[0]
	u_id = record[1]
	label = int(record[2])
	if i == 0:
		feats = np.hstack((user_info[u_id], ques_info[q_id]))
		if label == 0:
			labels = np.array([1, 0])
		else:
			labels = np.array([0, 1])
	else:
		curr_feats = np.hstack((user_info[u_id], ques_info[q_id]))
		if label == 0:
			curr_label = np.array([1, 0])
		else:
			curr_label = np.array([0, 1])
		feats = np.vstack((feats, curr_feats))
		labels = np.vstack((labels, curr_label))

cPickle.dump(feats, open('nn_features150.p', 'wb'))
cPickle.dump(labels, open('labels150.p', 'wb'))
print feats.shape, labels.shape
for i in range(len(val)):
	record = val[i]
	record = record.replace('\n', '').replace('\r', '')
	record = record.split(',')
	q_id = record[0]
	u_id = record[1]
	if i == 0:
		vfeats = np.hstack((user_info[u_id], ques_info[q_id]))
	else:
		vcurr_feats = np.hstack((user_info[u_id], ques_info[q_id]))
		vfeats = np.vstack((vfeats, vcurr_feats))

cPickle.dump(vfeats, open('nn_val_features150.p', 'wb'))



#print feats.shape, labels.shape
print vfeats.shape

for i in range(len(test)):
	record = test[i]
	record = record.replace('\n', '').replace('\r', '')
	record = record.split(',')
	q_id = record[0]
	u_id = record[1]
	if i == 0:
		tfeats = np.hstack((user_info[u_id], ques_info[q_id]))
	else:
		tcurr_feats = np.hstack((user_info[u_id], ques_info[q_id]))
		tfeats = np.vstack((tfeats, tcurr_feats))

cPickle.dump(tfeats, open('nn_test_features150.p', 'wb'))

print tfeats.shape



