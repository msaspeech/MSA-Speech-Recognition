from utils import int_to_binary, get_empty_binary_vector, convert_word_to_binary, convert_to_int, load_pickle_data, get_files_full_path
from tensorflow.python.keras import models
from lib import AudioInput
import numpy as np
from etc import settings

general_info = load_pickle_data(settings.DATASET_WORD_INFORMATION_PATH)
settings.MFCC_FEATURES_LENGTH = general_info[0]
settings.TOTAL_SAMPLES_NUMBER = general_info[1]
settings.WORD_SET = general_info[2]
settings.LONGEST_WORD_LENGTH = general_info[3]
settings.CHARACTER_SET = general_info[4]
settings.WORD_TARGET_LENGTH = general_info[5]


word_set_length = len(settings.WORD_SET)
cpt = 0
for word in settings.WORD_SET:
    if len(word) <= 7:
        cpt +=1

print(cpt)

per = int(cpt * 100 / word_set_length)
print(per)

