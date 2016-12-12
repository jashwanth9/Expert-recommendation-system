import json
import numpy as np
import cPickle as pickle
from scipy import sparse
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer


''' reads file and create a dictonary that maps an id with
    the related fields '''
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


def get_all_val_col(values, id_index):
    val = []
    for i in values:
        val = val + i[id_index]
    return val


def build_graph(info_data, info_keys):
	max_val = np.max(get_all_val_col(info_data.values(), 0)) + 1
	max_val1 = np.max(get_all_val_col(info_data.values(), 2)) + max_val + 1 
	print max_val
	with open("user_attribute.txt" , 'w') as f1:
		for i in info_keys:
			l1 = info_data[i][0]
			for k in l1:
				f1.write(i+' '+str(k)+'\n')
			l2 = info_data[i][2]
			for k in l2:
				f1.write(i+' '+str(max_val+k)+'\n')
			l3 = info_data[i][1]
			for k in l3:
				f1.write(i+' '+str(max_val1+k)+'\n')



if __name__ == '__main__':
	user_info_data, user_info_keys = read_files('../train_data/user_info.txt')
	# question_info_data, question_info_keys = read_files('../train_data/question_info.txt')
	# print user_info_data[user_info_keys[0]][0]
	build_graph(user_info_data, user_info_keys)
	print "done"