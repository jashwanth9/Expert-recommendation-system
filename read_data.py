import json
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from collections import Counter


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

'''reads invited_info_train.txt and store the question_id expert_id mapping'''
def read_invited_info():
    with open('invited_info_train.txt') as question_info_file:
        content = question_info_file.readlines()
    formated_content = {}
    values = []
    for i in content:
        element = i.strip("\n").split("\t")
        values.append(map(int, element[2].strip()))
        formated_content[element[0].strip() + " " +element[1].strip()] = values
    return formated_content

def get_all_val_col(values):
    val = []
    for i in values:
        val = val + i[0]
    return val

def user_print_to_csv(data, _keys, file_name):
    word_vec = pickle.load(open('userwordvec-1.p', 'rb'))
    # word_vec[np.isinf(word_vec)] = 0
    max_val = np.max(get_all_val_col(data.values()))
    new_dic = {}
    cou = 0
    for each_key in _keys:
        zer_vec = np.zeros((max_val))
        for content in data[each_key][0]:
            zer_vec[content-1] = 1
        new_dic[each_key] = np.hstack((zer_vec, word_vec[cou, :]))
        cou +=1
    pickle.dump(new_dic, open(file_name, "wb"))

def question_print_to_csv(data, _keys, file_name):
    word_vec = pickle.load(open('queswordvec-1.p', 'rb'))
    # word_vec[np.isinf(word_vec)] = 0
    max_val = np.max(get_all_val_col(data.values()))
    new_dic = {}
    cou = 0
    for each_key in _keys:
        zer_vec = np.zeros((max_val+3))
        for content in data[each_key][0]:
            zer_vec[content-1] = 1
        zer_vec[max_val] = data[each_key][3][0]
        zer_vec[max_val+1] = data[each_key][4][0]
        zer_vec[max_val+2] = data[each_key][5][0]
        new_dic[each_key] = np.hstack((zer_vec, word_vec[cou, :]))
        cou +=1
    pickle.dump(new_dic, open(file_name, "wb"))

def print_to_file(data, file_name):
    outfile = open(file_name, 'w')
    for i in data:
        line_str = ""
        for word in i[1]:
            line_str += str(word) + " "
        line_str = line_str[:-1] + '\n'  
        outfile.write(line_str)
    outfile.close()


if __name__ == "__main__":
    user_info_data, user_info_keys = read_files('user_info.txt')
    question_info_data, question_info_keys = read_files('question_info.txt')
    invited_info_train_data = read_invited_info()
    # print_to_file(user_info_data, 'user_info_character_id.txt')
    user_print_to_csv(user_info_data, user_info_keys, 'user_info_csv.dat')
    question_print_to_csv(question_info_data, question_info_keys, 'question_info_csv.dat')


    
