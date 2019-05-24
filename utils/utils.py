import os
import pickle
import re


def read_file_content(file_path):
    file_lines = []
    with open(file_path, "r") as file:
        file_lines = file.readlines()

    return file_lines


def get_files(directory):
    list_files = []
    for root, dirs, files in os.walk(directory):
        for filename in sorted(files):
            list_files.append(directory + filename)
        list_files.sort(key=natural_keys)
    return list_files


def file_exists(file_path):
    if os.path.exists(file_path):
        return True
    return False



def empty_directory(directory_path):
    if not os.listdir(directory_path):
        return True
    return False


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def int_to_binary(num):
    if num == 0:
        return [num]

    binary_digits = []
    while num != 0:
        modnum = int(num % 2)
        num = int(num / 2)
        binary_digits.append(modnum)
    return list(reversed(binary_digits))


def convert_word_to_binary(word_index, output_binary_vector):
    binary_value = int_to_binary(word_index)
    input_length = len(binary_value)-1
    output_length = len(output_binary_vector)-1

    for i in range(0, len(binary_value)):
    #for i, value in enumerate(binary_value):
        output_binary_vector[output_length-i] = binary_value[input_length-i]

    return output_binary_vector


def get_empty_binary_vector(upper_bound):
    binary_vector = int_to_binary(upper_bound)
    for i in range(0, len(binary_vector)):
        binary_vector[i] = 0
    return binary_vector


def generate_pickle_file(data, file_path):
    """
     Uploads AudioInput data from pickle file
     :return:
     """
    with open(file_path , 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def load_pickle_data(file_path):
    """
         Uploads AudioInput data after padding from pickle file
         :return:
         """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

