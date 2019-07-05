from utils import int_to_binary, get_empty_binary_vector, convert_word_to_binary, convert_to_int, load_pickle_data, get_files_full_path
from tensorflow.python.keras import models
from lib import AudioInput
import numpy as np
from etc import settings
import sys
from utils import buckwalter_to_arabic, arabic_to_buckwalter
from utils import load_pickle_data, generate_pickle_file

print(arabic_to_buckwalter("مشاهدينا الكرام السلام عليكم"))
print(buckwalter_to_arabic("mwqfy Alrsmy ywm Alsbt wsykwn fy hnAk"))

#test()