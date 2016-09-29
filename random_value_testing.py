#Script to assing random uniform probabilities to question,user pairs in validate_nolabel.txt
#This can be treated as baseline
import random

f2 = open('validation_random.csv', 'w')
with open('validate_nolabel.txt', 'r') as f1:
	header = f1.readline()
	f2.write(header)
	for line in f1:
		f2.write(line.rstrip('\r\n') + ',' + str(random.random()) + '\n')
	f2.close()

