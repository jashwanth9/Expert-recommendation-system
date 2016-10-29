import json
import numpy as np
import cPickle as pickle
from collections import Counter
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


def get_all_val_col(values, id_index):
    val = []
    for i in values:
        val = val + i[id_index]
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

'''create a max vector '''
def build_tfidf_que_word(data, _keys, id_index, file_name):
    max_val = np.max(get_all_val_col(data.values(), id_index))
    len_keys = len(_keys)
    zer_vec = np.zeros((len_keys, max_val))
    for cou in range(len_keys):
        for content in data[_keys[cou]][id_index]:
            zer_vec[cou][content-1] += 1
    print("done creating vector")

    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(zer_vec.tolist())
    pickle.dump(tfidf, open(file_name, "wb"))
    print("done TfidfTransformer")

    # tfidf_dict = {}
    # for cou in range(len_keys):
    #     tfidf_dict[_keys[cou]] = tfidf[cou].toarray()
    # pickle.dump(tfidf_dict, open(file_name, "wb"))


def compute_similarity(file_name):
    word_vec = pickle.load(open(file_name, 'rb'))
    A_sparse = sparse.csr_matrix(samples)
    similarities = cosine_similarity(A_sparse)


if __name__ == "__main__":
    id_index = 2

    user_info_data, user_info_keys = read_files('user_info.txt')
    question_info_data, question_info_keys = read_files('question_info.txt')
    print(np.max(get_all_val_col(user_info_data.values(), id_index)))
    print(np.max(get_all_val_col(question_info_data.values(), id_index)))
    invited_info_train_data = read_invited_info()
    
    print question_info_data[question_info_keys[0]][id_index]
    # build_tfidf_que_word(user_info_data, user_info_keys, id_index, 'user_charid_tfidf.dat')
    
    
    
