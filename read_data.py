import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import json

''' reads file and create a dictonary that maps an id with
    the related fields '''
def read_files(file_name):
    with open(file_name) as question_info_file:
        content = question_info_file.readlines()
    formated_content = {}
    for i in content:
        element = i.strip("\n").split("\t")
        values = []
        for i in range(1, len(element)):
            temp_element = element[i].strip()
            if temp_element == '/' or temp_element == '':
                values.append([])
            else:
                values.append(map(int, temp_element.split('/')))
        formated_content[element[0]] = values
    return formated_content

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

def plot_user_info_graphs(user_info_data):
    pass

def plot_question_info_data_graphs(question_info_data):
    pass

def print_word_id_to_file(data, file_name):
    val = []
    for i in data:
        val = val + i[2] 
    tup = Counter(val)
    with open(file_name, 'w') as outfile:
        json.dump(tup, outfile) 
    # plt.hist(val,max(val))
    # plt.title("Histogram")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.show()
        

if __name__ == "__main__":
    user_info_data = read_files('user_info.txt')
    question_info_data = read_files('question_info.txt')
    invited_info_train_data = read_invited_info()
    # print_word_id_to_file(user_info_data.values(), 'some.txt')


    # plot_user_info_graphs(user_info_data)
    # plot_question_info_data_graphs(question_info_data)
    
