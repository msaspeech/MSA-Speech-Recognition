from etc import settings
from lib import AudioInput
from utils import convert_to_int, convert_int_to_char, load_pickle_data, buckwalter_to_arabic
import numpy as np
from tensorflow.python.keras import models
from models import correct_word

def predict_sequence_test(audio_input):
    char_to_int = convert_to_int(sorted(settings.CHARACTER_SET))
    int_to_char = convert_int_to_char(char_to_int)

    t_force = "\tAlm$Akl Alty"
    encoded_transcript = []
    for index, character in enumerate(t_force):
        encoded_character = [0] * len(settings.CHARACTER_SET)
        position = char_to_int[character]
        encoded_character[position] = 1
        encoded_transcript.append(encoded_character)

    decoder_input = np.array([encoded_transcript])
    print(decoder_input.shape)
    model = models.load_model("model.h5")
    output = model.predict([audio_input, decoder_input])
    print(output.shape)
    sentence = ""
    output = output[0]
    for character in output:
        position = np.argmax(character)
        character = int_to_char[position]
        sentence += character

    print(sentence)
    print(buckwalter_to_arabic(sentence))
    corrected_sentence = []
    words = sentence.split()
    for word in words:
        corrected_word =correct_word(word)
        corrected_sentence.append(corrected_word)

    sentence = " ".join(corrected_sentence)
    print(sentence)
    print(buckwalter_to_arabic(sentence))

general_info = load_pickle_data("info_char.pkl")
settings.MFCC_FEATURES_LENGTH = general_info[0]
settings.CHARACTER_SET = general_info[2]
general_info = load_pickle_data("info_word.pkl")
settings.WORD_SET = general_info[2]

sample = AudioInput("speech.wav", "")
audio = [sample.mfcc.transpose()]

audio_sequence = np.array(audio, dtype=np.float32)
print(audio_sequence.shape)

predict_sequence_test(audio_sequence)
