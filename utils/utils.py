import os
import pickle


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

    return list_files


def generate_pickle_file(data, file_path):
    """
     Uploads AudioInput data from pickle file
     :return:
     """
    with open(file_path , 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def upload_pickle_data(file_path):
    """
         Uploads AudioInput data after padding from pickle file
         :return:
         """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

