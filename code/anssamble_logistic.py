import numpy as np
import cPickle as pickle
from sklearn import linear_model

def read_files(file_name):
    with open(file_name) as question_info_file:
        content = question_info_file.readlines()
    formated_content = {}
    list_keys = []
    for i in content:
        element = i.strip("\n").split("\t")
        values = []
        for i in range(1, len(element)):
            temp_element = element[i].strip()
            if temp_element == '/' or temp_element == '':
                values.append([])
            else:
                values.append(map(int, temp_element.split('/')))
        list_keys.append(element[0])
        formated_content[element[0]] = values
    return formated_content, list_keys


def norm_last3_feats(question_info_data, question_info_keys):
	x_train = []
	for i in question_info_keys:
		p = question_info_data[i]
		x_train.append(p[3] + p[4] + p[5])
	mean_train = np.mean(x_train, axis=0)
	std_train = np.std(x_train, axis=0)
	x_train = (x_train - mean_train)/(std_train)
	last3_feats = {}
	for i in question_info_keys:
		p = question_info_data[i]
		x_t = p[3] + p[4] + p[5]
		last3_feats[i] = (x_t - mean_train)/(std_train)
	return last3_feats


def loadTrainingData(inpFile, labelFile, ques_3_data):
	with open(labelFile) as invited_info_file:
		content = invited_info_file.readlines()
	with open(inpFile) as inp_info_file:
		train_data = inp_info_file.readlines()

	x_train = []
	y_train = []
	for i in range(len(content)):
		element = content[i].strip("\n").split("\t")
		inpele = train_data[i].strip("\n").split(",")
		x_train.append(np.hstack((ques_3_data[inpele[0]], np.array(float(inpele[2].strip())))))
		y_train.append(int(element[2].strip()))
	return x_train, y_train


def loadTestData(testFile, labelFile, ques_3_data):
	with open(labelFile) as invited_info_file:
		content = invited_info_file.readlines()
	testData = []
	with open(testFile, 'r') as f1:
		for line in f1:
			testData.append(line.rstrip('\r\n').split(','))
	x_test = []
	for i in range(len(content)):
		element = content[i].strip("\n").split(",")
		x_test.append(np.hstack((ques_3_data[element[0]], np.array(float(element[2].strip())))))
	return x_test, testData


def writeToOutputFile(x_test, y_pred, outFile, testData):
	with open(outFile, 'w') as f1:
		f1.write("qid,uid,label\n")
		y_pred[y_pred<0] = 0
		for i in range(len(x_test)):
			f1.write("{0},{1},{2}\n".format(testData[i+1][0], testData[i+1][1], y_pred[i][1]))


if __name__ == '__main__':
	question_info_data, question_info_keys = read_files('../train_data/question_info.txt')
	# p = question_info_data[question_info_keys[0]]
	# print p[3] + p[4] + p[5]
	ques_3_data = norm_last3_feats(question_info_data, question_info_keys)
	print ques_3_data[question_info_keys[0]]
	x_train, y_train = loadTrainingData('../features/contentbased_char_tfidfrevtrain.csv', '../train_data/invited_info_train.txt', ques_3_data)
	x_test, testData = loadTestData('../train_data/validate_nolabel.txt', '../features/content_char_tfidf_rev.csv', ques_3_data)
	print('x_train = ', x_train[0])
	print('y_train = ', y_train[0])
	print('x_test = ', x_test[0])
	lr = linear_model.LogisticRegression()

	model = lr.fit(x_train, y_train)
	print model.coef_
	print model.intercept_
	print model.get_params()
	y_pred = model.predict_proba(x_test)
	writeToOutputFile(x_test, y_pred, '../validation/val_anssamble_logreg.csv', testData)


# weights - [ 5.17249459]]   bias [-2.8807507]