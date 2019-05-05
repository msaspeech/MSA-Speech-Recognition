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


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def generate_pickle_file(data, file_path):
    """
     Uploads AudioInput data from pickle file
     :return:
     """
    with open(file_path , 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle_data(file_path):
    """
         Uploads AudioInput data after padding from pickle file
         :return:
         """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

