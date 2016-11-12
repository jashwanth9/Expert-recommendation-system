import json
import numpy as np
import cPickle as pickle
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


'''reads invited_info_train.txt and store the question_id expert_id mapping'''
def write_invited_info(question_info_data, user_info_data):
    with open('../train_data/invited_info_train.txt') as question_info_file:
        content = question_info_file.readlines()
    
    f0 = open("label0.csv", "w")
    f1 = open("label1.csv", "w")
    f0.write("sep=;\n")
    f1.write("sep=;\n")
    f0.write("upvotes_count;answers_count;top_quality_ans;ques_tag;user_tags;ques_word_id;user_word_id;ques_char_id;user_char_id\n")
    f1.write("upvotes_count;answers_count;top_quality_ans;ques_tag;user_tags;ques_word_id;user_word_id;ques_char_id;user_char_id\n")
    for i in content:
        element = i.strip("\n").split("\t")
        val = int(element[2].strip())
        p = question_info_data[element[0].strip()]
        q = user_info_data[element[1].strip()]
        if val == 1:
            f1.write("{0};{1};{2};{3};{4};{5};{6};{7};{8}\n".format(p[3], p[4], p[5], p[0], q[0], p[1], q[1], p[2], q[2]))
        else:
            f0.write("{0};{1};{2};{3};{4};{5};{6};{7};{8}\n".format(p[3], p[4], p[5], p[0], q[0], p[1], q[1], p[2], q[2]))
    f0.close()
    f1.close()


def get_all_val_col(values, id_index):
    val = []
    for i in values:
        val = val + i[id_index]
    return val


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
    id_index = 1

    user_info_data, user_info_keys = read_files('../train_data/user_info.txt')
    question_info_data, question_info_keys = read_files('../train_data/question_info.txt')
    print(question_info_data.values()[0])
    print(user_info_data.values()[0])
    write_invited_info(question_info_data, user_info_data)



