from etc import settings
from lib import AudioInput
from utils import convert_to_int, convert_int_to_char
import numpy as np
from tensorflow.python.keras import models

def predict_sequence_test(audio_input):
    char_to_int = convert_to_int(sorted(settings.CHARACTER_SET))
    int_to_char = convert_int_to_char(char_to_int)

    t_force = "SOS_ m$AhdynA AlkrAm AlslAm Elykm _EOS"
    words = t_force.split()
    # print(words)
    character_set_length = 50
    dec_input_len = 800
    encoded_transcript = []
    for word in words:
        encoded_word = [0] * dec_input_len
        i = 0
        for i, character in enumerate(word):
            position = character_set_length * i + char_to_int[character]
            encoded_word[position] = 1
            # print(position)

        if i < 16 - 1:
            for k in range(i + 1, 16):
                position = character_set_length * k + character_set_length - 1
                encoded_word[position] = 1

        encoded_transcript.append(encoded_word)

    transcript = np.zeros((1, len(words), dec_input_len))
    transcript[0] = encoded_transcript

    # print(transcript.shape)
    # print(transcript[0][0])
    # print([np.argmax(transcript[0][0][i:i+41]) for i in range(0, 574, 41)])
    model = models.load_model("model.h5")
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


sample = AudioInput("audio.wav", "")
audio = [sample.mfcc.transpose()]

audio_sequence = np.array(audio, dtype=np.float32)
print(audio_sequence.shape)

predict_sequence_test(audio_sequence)
