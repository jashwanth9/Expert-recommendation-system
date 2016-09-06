import matplotlib.pyplot as plt
import numpy as np


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
    for i in content:
        element = i.strip("\n").split("\t")
        values.append(map(int, element[2].strip()))
        formated_content[element[0].strip() + " " +element[1].strip()] = values
    return formated_content

def plot_user_info_graphs(user_info_data):
    pass

def plot_question_info_data_graphs(question_info_data):
    pass

def plot_data(graph_data):
    n, bins, patches = plt.hist(graph_data, max(graph_data),
                                facecolor='g', alpha=0.75)
    plt.xlabel('dont know ')
    plt.ylabel('dont know')
    plt.title('Histogram of Expert user tags')
    plt.axis([min(graph_data), max(graph_data), 0, 3500])
    # need to find a way to figure out the scale of data ie 3500
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    user_info_data = read_files('user_info.txt')
    question_info_data = read_files('question_info.txt')
    # plot_user_info_graphs(user_info_data)
    # plot_question_info_data_graphs(question_info_data)
    invited_info_train_data = read_invited_info()
