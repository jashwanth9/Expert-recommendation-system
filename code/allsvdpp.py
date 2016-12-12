#from __future__ import print_function
import random
import random_value_testing
import numpy as np
import content_based_tags
import collabFiltering_cross
import pymp
import copy
import dimreductionCollab
import collab_content_based_tags
import os



def cv(k):
	folds = 8

	trainData = []
	with open('../train_data/invited_info_train.txt', 'r') as f1:
			for line in f1:
				line = line.rstrip('\n')
				sp = line.split()
				trainData.append((sp[1], sp[0], int(sp[2])))
	random.shuffle(trainData)
	N = len(trainData)
	#res = pymp.shared.array((folds,), dtype='uint8')
	#rp = []
	


	rp = pymp.shared.list()
	with pymp.Parallel(8) as p:
		r = 0
		for i in p.range(8):
			td = trainData[:(i)*(N/folds)] + trainData[(i+1)*(N/folds):]
			tdfname = 'cv/train'+str(i)
			with open(tdfname, 'w') as f1:
				for e in range(len(td)):
					f1.write(td[e][0]+' '+td[e][1]+' '+str(td[e][2])+'\n')
			valData = trainData[i*(N/folds):(i+1)*(N/folds)]
			valfname = 'cv/val'+str(i)
			with open(valfname, 'w') as f1:
				for e in range(len(valData)):
					f1.write(valData[e][0]+','+valData[e][1]+','+str(valData[e][2])+'\n')

			#print len(td)
			#print len(valData)
			fname = 'cv/allGSVDPlusPlus' + str(i)
			os.system('../MyMediaLite-3.10/bin/rating_prediction --training-file='+tdfname+' --recommender=GSVDPlusPlus --recommender-options="num_factors=50 regularization=0.015 bias_reg=0.33 frequency_regularization=False learn_rate=0.01 bias_learn_rate=0.7 num_iter=30 num_iter=200" --item-attributes=../MyMediaLite-3.10/item_attribute.txt --user-attributes=../MyMediaLite-3.10/user_attribute.txt --prediction-line="{1},{0},{2}" --prediction-header="qid,uid,label" --test-file='+valfname+' --prediction-file='+fname)

			r = evaluate.ndcg(fname)
			#r =  random_value_testing.run(td, valData, i, k)
			# r = dimreductionCollab.run(td, valData, k, i)
			# r = collab_content_based_tags.run(td, valData)
			print r
			#r = collabFiltering_cross.run(td, valData, i, k)
			with p.lock:
			 	rp.append(r)
		print rp
		print "Mean"
		print np.mean(rp)



for k in range(1, 2):
	print k
	cv(k)














