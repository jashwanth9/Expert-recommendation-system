#takes in a file and reverses the order of qid and uid


import sys
f2 = open(sys.argv[2], 'w')
with open(sys.argv[1], 'r') as f1:
	f2.write(f1.readline())
	for line in f1:
		#print line
		#print "h"
		#qid, uid, val = line.rstrip('\n').split()
		qid, uid = line.rstrip('\r\n').split(',')
		#nval = (float(val1) + float(val2))/2
		f2.write(uid+','+qid+'\n')
		#f2.write(uid+' ' +qid+' '+val+'\n')
f2.close()
