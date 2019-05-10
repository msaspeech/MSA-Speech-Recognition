import os
import pickle
import re
from num2words import num2words
from lang_trans.arabic import buckwalter


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


def arabic_to_buckwalter(arabic_sentence):
    return buckwalter.transliterate(arabic_sentence)


def buckwalter_to_arabic(buckwalter_sentence):
    return buckwalter.untransliterate(buckwalter_sentence)


def convert_numeral_to_written_numbers(number):
    written_number = num2words(number, lang="ar")
    return written_number


def numerical_to_written_numbers_table(inf_number=0, sup_number=10000):
    mapped_numbers = {}
    for i in range(inf_number, sup_number):

        mapped_numbers[i] = arabic_to_buckwalter(convert_numeral_to_written_numbers(i))
    return mapped_numbers


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
