# gets word/characters usefulness (described in paper: I Want to Answer, Who has a Question?)
import pdb
import math
import operator
import matplotlib.pyplot as plt

def getUscores(tcounts, T):
	alpha = 20/float(T)
	Lt = {}
	for term in tcounts:
		Lt[term] = 0
		for c in range(T):
			prc = 0
			den = 0
			for c2 in range(T):
				den += tcounts[term][c2] + alpha
			prc = (tcounts[term][c]+alpha)/float(den)
			Lt[term] -= (prc*math.log(prc))
	return Lt





wordcount = {}
charcount = {}
best = 0
with open('../train_data/question_info.txt', 'r') as f1:

	for line in f1:
		sp = line.split()
		topic = int(sp[1])
		#best = max(topic, best)
		try:
			words = map(int, sp[2].split('/'))
		except:
			words = []
		try:
			chars = map(int, sp[3].split('/'))
		except:
			chars = []
		for word in words:
			if word not in wordcount:
				wordcount[word] = [0]*20
			wordcount[word][topic] += 1
		for char in chars:
			if char not in charcount:
				charcount[char] = [0]*20
			charcount[char][topic] += 1

Lc = getUscores(charcount, 20)
sorted_x = sorted(Lc.items(), key=operator.itemgetter(1))
print sorted_x[:20]
print len(sorted_x)
plt.plot([x[1] for x in sorted_x], range(len(sorted_x)))
plt.show()
Lw = getUscores(wordcount, 20)
sorted_x = sorted(Lw.items(), key=operator.itemgetter(1))
plt.plot([x[1] for x in sorted_x], range(len(sorted_x)))
plt.show()