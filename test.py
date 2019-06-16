from utils import int_to_binary, get_empty_binary_vector, convert_word_to_binary, convert_to_int, load_pickle_data, get_files_full_path
from tensorflow.python.keras import models
from lib import AudioInput
import numpy as np
from etc import settings
import sys

from utils import load_pickle_data, generate_pickle_file

dataset1 = load_pickle_data("./data/partitions/dataset0.pkl")
dataset2 = load_pickle_data("./data/partitions/dataset1.pkl")
dataset3 = load_pickle_data("./data/partitions/dataset2.pkl")
dataset4 = load_pickle_data("./data/partitions/dataset3.pkl")
dataset5 = load_pickle_data("./data/partitions/dataset4.pkl")


dataset = dataset1 + dataset2 + dataset3 + dataset4 + dataset5

generate_pickle_file(dataset, "./data/partitions/dataset0.pkl")


