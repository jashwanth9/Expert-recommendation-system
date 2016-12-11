f = open('bytecup2016data/invited_info_train.txt', 'r')
labels = f.readlines()
f.close()
f = open('bytecup2016data/user_info.txt', 'r')
user_info = f.readlines()
f.close()
f = open('bytecup2016data/question_info.txt', 'r')
q_info = f.readlines()
f.close()

f = open('user_word_ids.txt', 'w')
f1 = open('user_char_ids.txt', 'w')
max_tid = -1
for data in user_info:
	data = data.split('\t')
	#topic ids
	tids = data[1]
	tid = max(map(int, tids.split('/')))
	if tid > max_tid:
		max_tid = tid
	word_seq = data[2]
	word_seq = word_seq.split('/')
	f.write(' '.join(word_seq) + '\n')
	char_seq = data[3]
	char_seq = char_seq.replace('\n', '').replace('\r', '')
	char_seq = char_seq.split('/')
	f1.write(' '.join(char_seq) + '\n')
f.close()
f1.close()

f = open('ques_word_ids.txt', 'w')
f1 = open('ques_char_ids.txt', 'w')
for data in q_info:
	data = data.split('\t')
	word_seq = data[2]
	word_seq = word_seq.split('/')
	f.write(' '.join(word_seq) + '\n')
	char_seq = data[3]
	char_seq = char_seq.replace('\n', '').replace('\r', '')
	char_seq = char_seq.split('/')
	f1.write(' '.join(char_seq) + '\n')
f.close()
f1.close()

print max_tid
