from collections import Counter


f1 = open('../features/question_char_freq.txt', 'w')

queswords = []
maxwordid = 1
with open('../train_data/question_info.txt', 'r') as f2:
	for line in f2:
		sp = line.split()
		try:
			words = map(int, sp[3].split('/'))
			maxwordid = max(maxwordid, max(words))
		except:
			words = []
		queswords.append(Counter(words))
		

for words in queswords:
	for i in range(maxwordid+1):
		if i in words:
			f1.write(str(words[i])+' ')
		else:
			f1.write('0 ')
	f1.write('\n')

