#takes 2 predictions files as command line input arguments and outputs a new predictions
#with the average of their outputs

import sys
f2 = open(sys.argv[2], 'r')
f3 = open(sys.argv[3], 'w')
with open(sys.argv[1], 'r') as f1:
	line = f1.readline()
	f3.write(line)
	f2.readline()
	for line in f1:
		#print line
		#print "h"
		qid, uid, val1 = line.rstrip('\n').split(',')
		val2 = f2.readline().rstrip('\n').split(',')[2]
		nval = (float(val1) + float(val2))/2
		f3.write(qid+','+uid+','+str(nval)+'\n')
f2.close()
f3.close()