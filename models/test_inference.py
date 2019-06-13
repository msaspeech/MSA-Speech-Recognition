from etc import settings
from utils import load_pickle_data
from tensorflow.python.keras import models
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input
from utils import convert_to_int, convert_int_to_char
from . import correct_word
import numpy as np
from utils import buckwalter_to_arabic

class Word_Inference_TEST():

    def __init__(self, model_path, latent_dim):
        self.model = models.load_model(model_path)
        self.model.summary()
        self.encoder_states = None
        self.latent_dim = latent_dim

        # Getting dataset and training information
        general_info = load_pickle_data(settings.DATASET_WORD_INFERENCE_INFORMATION_PATH)
        settings.MFCC_FEATURES_LENGTH = general_info[0]
        settings.TOTAL_SAMPLES_NUMBER = general_info[1]
        settings.WORD_SET = general_info[2]
        settings.LONGEST_WORD_LENGTH = general_info[3]
        settings.CHARACTER_SET = general_info[4]
        settings.WORD_TARGET_LENGTH = general_info[5]

        self.encoder_model = None
        self.decoder_model = None
        self.get_encoder_decoder_baseline_test()

    def test_encoder_decoder(self, audio_input):
        char_to_int = convert_to_int(sorted(settings.CHARACTER_SET))
        int_to_char = convert_int_to_char(char_to_int)

        encoder_inputs = Input(shape=(None, settings.MFCC_FEATURES_LENGTH))
        encoder_gru = self.model.get_layer("encoder_gru_layer")
        encoder_output, h = encoder_gru(encoder_inputs)
        self.encoder_states = h

        encoder_model = Model(encoder_inputs, self.encoder_states)

        state_h = encoder_model.predict(audio_input)
        print(state_h)


    def predict_sequence_test(self, audio_input):
        char_to_int = convert_to_int(sorted(settings.CHARACTER_SET))
        int_to_char = convert_int_to_char(char_to_int)

        t_force = "SOS_ msA' Alxyr >wlA hw Almwqf tm tSHyHh _EOS"
        words = t_force.split()
        #print(words)
        character_set_length = len(settings.CHARACTER_SET) + 1
        dec_input_len = character_set_length * settings.LONGEST_WORD_LENGTH
        encoded_transcript = []
        for word in words:
            encoded_word = [0]*dec_input_len
            i = 0
            for i, character in enumerate(word):
                position = character_set_length* i + char_to_int[character]
                encoded_word[position] = 1
                #print(position)

            if i < settings.LONGEST_WORD_LENGTH - 1:
                for k in range(i + 1, settings.LONGEST_WORD_LENGTH):
                    position = character_set_length * k + character_set_length - 1
                    encoded_word[position] = 1

            encoded_transcript.append(encoded_word)

        transcript = np.zeros((1,len(words),dec_input_len))
        transcript[0] = encoded_transcript

        #print(transcript.shape)
        #print(transcript[0][0])
        #print([np.argmax(transcript[0][0][i:i+41]) for i in range(0, 574, 41)])
        result = self.model.predict([audio_input, transcript])

        list_words = [""] * len(words)
        for i, characters in enumerate(result):
            encoded_characters= characters[0]
            for i, encoded_character in enumerate(encoded_characters):
                index = np.argmax(np.array(encoded_character))
                if index == character_set_length - 1:
                    character = ""
                else:
                    character = int_to_char[index]
                list_words[i] += character

        print(list_words)

    def get_encoder_decoder_baseline_test(self):

        # Getting encoder model
        encoder_inputs = self.model.get_layer("encoder_input").input
        encoder_gru = self.model.get_layer("encoder_gru_layer")
        encoder_output, h = encoder_gru(encoder_inputs)
        self.encoder_states = h

        self.encoder_model = Model(encoder_inputs, self.encoder_states)
        self.encoder_model.summary()
        # Getting decoder model

        decoder_inputs = self.model.get_layer("decoder_input").input

        decoder_gru1_layer = self.model.get_layer("decoder_gru1_layer")
        decoder_gru2_layer = self.model.get_layer("decoder_gru2_layer")
        decoder_gru3_layer = self.model.get_layer("decoder_gru3_layer")
        decoder_gru4_layer = self.model.get_layer("decoder_gru4_layer")

        decoder_dense_layers = []
        for i in range(0, settings.LONGEST_WORD_LENGTH):
            layer_name = "decoder_dense"+str(i)
            decoder_dense_layers.append(self.model.get_layer(layer_name))

        decoder_state_input_h = Input(shape=(self.latent_dim, ))
        decoder_states_inputs = [decoder_state_input_h]

        decoder_gru1, state_h = decoder_gru1_layer(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_gru2 = decoder_gru2_layer(decoder_gru1)
        decoder_gru3 = decoder_gru3_layer(decoder_gru2)
        decoder_output = decoder_gru4_layer(decoder_gru3)

        print(decoder_gru1_layer.output)

        decoder_states = [state_h]
        # getting dense layers as outputs
        decoder_outputs = []
        for i in range(0, settings.LONGEST_WORD_LENGTH):
            output = decoder_dense_layers[i](decoder_output)
            decoder_outputs.append(output)

        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            decoder_outputs + decoder_states)

    def get_encoder_decoder_baseline(self):

        # Getting encoder model
        encoder_inputs = self.model.get_layer("encoder_input").input
        encoder_gru = self.model.get_layer("encoder_gru_layer")
        encoder_output, h = encoder_gru(encoder_inputs)
        self.encoder_states = h

        self.encoder_model = Model(encoder_inputs, self.encoder_states)
        self.encoder_model.summary()
        # Getting decoder model

        decoder_inputs = self.model.get_layer("decoder_input").input

        decoder_gru1_layer = self.model.get_layer("decoder_gru1_layer")
        print(decoder_gru1_layer.output)
        decoder_gru2_layer = self.model.get_layer("decoder_gru2_layer")
        decoder_gru3_layer = self.model.get_layer("decoder_gru3_layer")
        decoder_gru4_layer = self.model.get_layer("decoder_gru4_layer")
        decoder_dense_layers = []
        for i in range(0, settings.LONGEST_WORD_LENGTH):
            layer_name = "decoder_dense"+str(i)
            decoder_dense_layers.append(self.model.get_layer(layer_name))

        decoder_state_input_h = Input(shape=(self.latent_dim, ))
        decoder_states_inputs = [decoder_state_input_h]

        decoder_gru1, state_h = decoder_gru1_layer(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_gru2, state_h2 = decoder_gru2_layer(decoder_gru1, initial_state=state_h)
        decoder_gru3, state_h3 = decoder_gru3_layer(decoder_gru2, initial_state=state_h2)
        decoder_output, state_h4 = decoder_gru4_layer(decoder_gru3, initial_state=state_h3)

        print(decoder_gru1_layer.output)

        decoder_states = [state_h, state_h2, state_h3, state_h4]
        # getting dense layers as outputs
        decoder_outputs = []
        for i in range(0, settings.LONGEST_WORD_LENGTH):
            output = decoder_dense_layers[i](decoder_output)
            decoder_outputs.append(output)

        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            decoder_outputs + decoder_states)

    def decode_audio_sequence(self, audio_sequence):

        # get layer (encoder ===> good model)

        # Getting converters
        char_to_int = convert_to_int(sorted(settings.CHARACTER_SET))
        int_to_char = convert_int_to_char(char_to_int)

        states_value = self.encoder_model.predict(audio_sequence)
        print("ENCODER PREDICTION DONE")

        # creating first input_sequence for decoder

        target_sequence = np.zeros((1, 1, settings.WORD_TARGET_LENGTH), dtype=np.float32)
        print(self.decoder_model.summary())
        sos_characters = ["S", "O", "S", "_"]
        target_length = len(settings.CHARACTER_SET) + 1
        for i in range(0, 4):
            position = char_to_int[sos_characters[i]] + i*target_length

            target_sequence[0, 0, position] = 1

        for i in range(4, settings.LONGEST_WORD_LENGTH):
            position = i * target_length + target_length - 1
            target_sequence[0, 0, position] = 1

        #print(target_sequence)
        stop_condition = False

        decoded_sentence = ""
        limit = 0
        while not stop_condition:
            #print("STATES VALUE IS ")
            #print(states_value)
            print("TARGET SEQUENCE IS ")
            print([np.argmax(target_sequence[0][0][i:i + 41]) for i in range(0, 574, 41)])
            #print(target_sequence)
            limit += 1
            result = self.decoder_model.predict([target_sequence, states_value], steps=1)

            dense_outputs = []
            for i in range(0, settings.LONGEST_WORD_LENGTH):
                dense_outputs.append(result[i])

            word = ""   

            print(word)
            h = result[-1]
            states_value = h
            print("DECODER PREDICTION DONE")

            # decoding values of each dense output
            decoded_word = ""

            for dense in dense_outputs:
                index = np.argmax(dense[0])
                if index == len(char_to_int):
                    decoded_word+=""
                else:
                    decoded_word+= int_to_char[index]

            print("decoded_word is : "+decoded_word)
            #corrected_word = correct_word(decoded_word)
            #print("corrected_word is : "+corrected_word)
            #print("corrected word in arabic is :"+buckwalter_to_arabic(corrected_word))
            decoded_sentence += decoded_word + " "
            if limit > 4:
                stop_condition = True

            if decoded_word == "EOS_":
                stop_condition = True
            else:
                target_sequence = np.zeros((1, 1, settings.WORD_TARGET_LENGTH))
                i = 0
                for i, character in enumerate(decoded_word):
                    position = char_to_int[character] + i * target_length
                    target_sequence[0, 0, position] = 1

                if i < settings.LONGEST_WORD_LENGTH - 1:
                    for j in range(i+1, settings.LONGEST_WORD_LENGTH):
                        position = i * target_length + target_length - 1
                        target_sequence[0, 0, position] = 1
