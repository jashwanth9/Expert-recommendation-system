import json
import numpy as np
import cPickle as pickle


with open('../validation/v_xgboost_word_tfidf.csv') as train_file:
        content = train_file.readlines()
testData = []
scores = []
element = content[1].strip("\r\n").split(",")
for i in range(1, len(content)):
    element = content[i].strip("\r\n").split(",")
    testData.append([element[0],element[1]])
    scores.append(float(element[2]))


predictions = []
maxscore = max(scores)
minscore = min(scores)
for score in scores:
        predictions.append((score-minscore)/float(maxscore-minscore))

ypred = predictions

with open('../validation/v_xgboost_word_tfidf_0-1.csv', 'w') as f1:
        f1.write('qid,uid,label\n')
        for i in range(0, len(ypred)):
                f1.write(testData[i][0]+','+testData[i][1]+','+str(ypred[i])+'\n')