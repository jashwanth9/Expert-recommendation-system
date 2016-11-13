#from __future__ import print_function
import random
import random_value_testing
import numpy as np
import content_based_tags
import collabFiltering_cross
import pymp
import copy




def cv(k):
	folds = 8

	trainData = []
	with open('../train_data/invited_info_train.txt', 'r') as f1:
			for line in f1:
				line = line.rstrip('\n')
				sp = line.split()
				trainData.append((sp[0], sp[1], int(sp[2])))
	random.shuffle(trainData)
	N = len(trainData)
	#res = pymp.shared.array((folds,), dtype='uint8')
	#rp = []
	rp = pymp.shared.list()
	with pymp.Parallel(8) as p:
		r = 0
		for i in p.range(folds):
			td = trainData[:(i)*(N/folds)] + trainData[(i+1)*(N/folds):]
			valData = [x[:2] for x in trainData[i*(N/folds):(i+1)*(N/folds)]]
			#print len(td)
			#print len(valData)

			#r = content_based_tags.run(td, valData)
			#r =  random_value_testing.run(td, valData, i, k)
			r = collabFiltering_cross.run(td, valData, i, k)
			with p.lock:
				rp.append(r)
			
	print rp
	with open('cvkcollab.txt', 'a') as f1:
		f1.write(str(k)+','+str(np.mean(rp))+'\n')


for k in range(10, 201, 10):
	print k
	cv(k)
