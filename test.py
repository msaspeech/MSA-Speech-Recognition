from utils import int_to_binary, get_empty_binary_vector, convert_word_to_binary, convert_to_int, load_pickle_data
from tensorflow.python.keras import models
from lib import AudioInput
import numpy as np
from etc import settings

infos = load_pickle_data(settings.DATASET_INFORMATION_PATH)
settings.CHARACTER_SET = infos[3]
char_to_int = convert_to_int(settings.CHARACTER_SET)

sample = AudioInput("test.wav", "")
print(sample.mfcc.transpose())
audio = [sample.mfcc.transpose()]
audio_sequence = np.array(audio)
target_sequence = np.zeros((1, 15, 882))
transcript = "SOS_ lr}Asp Aljmhwryp fy syAq AlAntxAbAt Alr}Asyp Almqblp lmA*A h*A AlqrAr fy Alwqt Al*y _EOS"
encoded_word = [0]*882
encoded_transcript = []
#target_sequence = np.array([None])
for word_index, word in enumerate(transcript.split()):
    #print(word_index)
    for index_character, character in enumerate(word):
        index = char_to_int[character] + (index_character * len(settings.CHARACTER_SET))
        target_sequence[0, word_index, index] = 1
        #encoded_word[index] = 1
    #encoded_transcript.append(encoded_word)


#target_sequence[0] = encoded_transcript
print(target_sequence.shape)

model = models.load_model("trained_models/architecture1.h5")
output = model.predict([audio, target_sequence])
print(output)
