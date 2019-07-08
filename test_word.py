from etc import settings
from lib import AudioInput
from utils import convert_to_int, convert_int_to_char, load_pickle_data, buckwalter_to_arabic, arabic_to_buckwalter
import numpy as np
from tensorflow.python.keras import models
from models import correct_word

def predict_sequence_test(audio_input):
    char_to_int = convert_to_int(sorted(settings.CHARACTER_SET))
    int_to_char = convert_int_to_char(char_to_int)

    t_force = "SOS_ kyf qmt b*lk _EOS"
    words = t_force.split()
    # print(words)
    character_set_length = len(settings.CHARACTER_SET) + 1
    dec_input_len = character_set_length * settings.LONGEST_WORD_LENGTH
    encoded_transcript = []
    for word in words:
        encoded_word = [0] * dec_input_len
        i = 0
        for i, character in enumerate(word):
            position = character_set_length * i + char_to_int[character]
            encoded_word[position] = 1
            # print(position)

        if i < settings.LONGEST_WORD_LENGTH - 1:
            for k in range(i + 1, settings.LONGEST_WORD_LENGTH):
                position = character_set_length * k + character_set_length - 1
                encoded_word[position] = 1

        encoded_transcript.append(encoded_word)

    transcript = np.zeros((1, len(words), dec_input_len))
    transcript[0] = encoded_transcript

    model = models.load_model("model_word.h5")
    result = model.predict([audio_input, transcript])

    list_words = [""] * len(words)
    for i, characters in enumerate(result):
        encoded_characters = characters[0]
        for i, encoded_character in enumerate(encoded_characters):
            index = np.argmax(np.array(encoded_character))
            if index == character_set_length - 1:
                character = ""
            else:
                character = int_to_char[index]
            list_words[i] += character

    print(list_words)
    corrected_sentence = []
    for word in list_words:
        if word != "_EOS":
            corrected_word = correct_word(word)
            corrected_sentence.append(corrected_word)

    sentence = " ".join(corrected_sentence)
    print(sentence)
    print(buckwalter_to_arabic(sentence))

general_info = load_pickle_data("info_word.pkl")
settings.MFCC_FEATURES_LENGTH = general_info[0]
settings.TOTAL_SAMPLES_NUMBER = general_info[1]
settings.WORD_SET = general_info[2]
settings.LONGEST_WORD_LENGTH = general_info[3]
settings.CHARACTER_SET = general_info[4]
settings.WORD_TARGET_LENGTH = general_info[5]

sample = AudioInput("speech.wav", "")
audio = [sample.mfcc.transpose()]

audio_sequence = np.array(audio, dtype=np.float32)
print(audio_sequence.shape)

predict_sequence_test(audio_sequence)
