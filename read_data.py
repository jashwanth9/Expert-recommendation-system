''' reads user_info.txt file and create a dictonary that maps 
    the expert id with the related fields '''
def read_user_info():
    with open('user_info.txt') as user_info_file:
        content = user_info_file.readlines()
    formated_content = {}
    for i in content:
        element = i.split()
        values = []
        for i in range(1, len(element)):
            values.append(element[i].split('/'))
        formated_content[element[0]] = values
    return formated_content

''' reads question_info.txt file and create a dictonary that maps 
    the expert id with the related fields '''
def read_question_info_data():
    with open('question_info.txt') as question_info_file:
        content = question_info_file.readlines()
    formated_content = {}
    for i in content:
        element = i.split()
        values = []
        for i in range(1, len(element)):
            values.append(element[i].split('/'))
        formated_content[element[0]] = values
    return formated_content

''' reads file and create a dictonary that maps 
    the expert id with the related fields '''
def read_files(file_name):
    with open(file_name) as question_info_file:
        content = question_info_file.readlines()
    formated_content = {}
    for i in content:
        element = i.split()
        values = []
        for i in range(1, len(element)):
            values.append(element[i].split('/'))
        formated_content[element[0]] = values
    return formated_content

if __name__ == "__main__":
    user_info_data = read_files('user_info.txt')
    question_info_data = read_files('question_info.txt')
    invited_info_train_data = read_files('invited_info_train.txt')
