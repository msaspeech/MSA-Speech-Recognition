from utils import int_to_binary, get_empty_binary_vector, convert_word_to_binary, convert_to_int, load_pickle_data, get_files_full_path
from tensorflow.python.keras import models
from lib import AudioInput
import numpy as np
from etc import settings


files = get_files_full_path(settings.AUDIO_SPLIT_TRAIN_PATH)
print(files)
x = 3
if x % 2 == 0:
    print("OUIIIIIIIIIIIIIIIIII")

x = [1, 3, 5, 15, 16]
i= 0
for i, elt in enumerate(x):
    print(elt)

print(i)

infos = load_pickle_data(settings.DATASET_INFORMATION_PATH)
settings.CHARACTER_SET = infos[3]
char_to_int = convert_to_int(settings.CHARACTER_SET)

sample = AudioInput("test.wav", "")
audio = [sample.mfcc.transpose()]
audio_sequence = np.array(audio)
target_sequence = np.zeros((1, 15, 882))
transcript = "SOS_ lr}Asp Aljmhwryp fy syAq AlAntxAbAt Alr}Asyp Almqblp lmA*A h*A AlqrAr fy Alwqt Al*y _EOS"
encoded_word = [0]*882
encoded_transcript = []
for word_index, word in enumerate(transcript.split()):
    for index_character, character in enumerate(word):
        index = char_to_int[character] + (index_character * len(settings.CHARACTER_SET))
        target_sequence[0, word_index, index] = 1

#print(target_sequence)
for word in target_sequence[0]:
    indices = []
    for index, character in enumerate(word):
        if character==1:
            indices.append(index)
    print(indices)

#print(target_sequence.shape)

model = models.load_model("trained_models/architecture1.h5")
output = model.predict([audio, target_sequence])
output = np.round(output)

print(output.shape)

#print(output)
for word in output[0]:
    indices = []
    for index, character in enumerate(word):
        if character==1:
            indices.append(index)
    print(indices)