import numpy as np
from utils import convert_to_int, convert_int_to_char, decode_transcript, load_pickle_data
from tensorflow.python.keras import models
from tensorflow.python.keras import Model
#import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input
from etc import settings
#from keras.layers import Input
import keras.backend as K
from .encoder_decoder import get_encoder_states


class Inference():

    def __init__(self, model_path, latent_dim, word_level=False):
        self.model = models.load_model(model_path)
        self.encoder_states = None
        self.latent_dim = latent_dim
        general_info = load_pickle_data(settings.DATASET_INFERENCE_INFORMATION_PATH)
        settings.MFCC_FEATURES_LENGTH = general_info[0]
        settings.CHARACTER_SET = general_info[2]
        self.character_set = settings.CHARACTER_SET
        self.word_level = word_level
        self.encoder_model, self.decoder_model = self._get_encoder_decoder_model_baseline()

    def _get_encoder_decoder_model_baseline(self):
        # Getting layers after training (updated weights)

        #encoder_inputs = Input(shape=(None, settings.MFCC_FEATURES_LENGTH))
        #decoder_inputs = Input(shape=(None, len(settings.CHARACTER_SET)))
        encoder_inputs = self.model.get_layer("encoder_input").input
        #print(self.model.get_layer("encoder_lstm_layer").output[0])
        [h, c] = self.model.get_layer("encoder_lstm_layer").output[0], self.model.get_layer("encoder_lstm_layer").output[1]
        self.encoder_states = [h, c]
        #print(self.encoder_states)
        decoder_inputs = self.model.get_layer("decoder_input").input
        decoder_lstm1_layer = self.model.get_layer("decoder_lstm1_layer")
        decoder_lstm2_layer = self.model.get_layer("decoder_lstm2_layer")
        decoder_dense = self.model.get_layer("decoder_dense")

        # getting_encoder_states

        # Creating encoder model
        #self.encoder_states = get_encoder_states(settings.MFCC_FEATURES_LENGTH, encoder_inputs=encoder_inputs, latent_dim=self.latent_dim)
        #print(self.encoder_states)
        encoder_model = Model(encoder_inputs, self.encoder_states)
        #encoder_model = K.function([encoder_inputs], [self.encoder_states])

        # Input shapes for 1st LSTM layer
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm1 = decoder_lstm1_layer(decoder_inputs, initial_state=decoder_states_inputs)

        # Outputs and states from final LSTM Layer
        decoder_outputs, state_h, state_c = decoder_lstm2_layer(decoder_lstm1)
        decoder_states = [state_h, state_c]

        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model

    def decode_audio_sequence_character_based(self, audio_sequence):
        """
        Decodes audio sequence into a transcript using encoder_model and decoder_model generated from training
        :param audio_sequence: 2D numpy array
        :param encoder_model: Model
        :param decoder_model: Model
        :param character_set: Dict
        :return: String
        """
        # Getting converters
        char_to_int = convert_to_int(sorted(settings.CHARACTER_SET))
        int_to_char = convert_int_to_char(char_to_int)
        print(char_to_int)
        print(int_to_char)

        # Returns the encoded audio_sequence
        states_value = self.encoder_model.predict(audio_sequence)
        print("ENCODER PREDICTION DONE")
        num_decoder_tokens = len(char_to_int)
        #target_character = np.zeros((1, 1, num_decoder_tokens))
        target_sequence = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first character of target sequence with the start character.
        target_sequence[0, 0, char_to_int['\t']] = 1.
        print(target_sequence)
        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_sequence] + states_value)

            print("DECODER PREDICTION DONE")
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = int_to_char[sampled_token_index]
            print(sampled_char)
            decoded_sentence += sampled_char

            if sampled_char == "\n":
                # End of transcription
                stop_condition = True
            else:
                # updating target sequence vector
                target_sequence = np.zeros((1, 1, num_decoder_tokens))
                target_sequence[0, 0, sampled_token_index] = 1

            states_values = [h, c]

        return decoded_sentence


    def measure_test_accuracy(self, test_data, transcripts, model, encoder_states, latent_dim):
        test_size = len(test_data)
        good_prediction = 0
        for i, audio_input in enumerate(test_data):
            predicted_sequence = self.predict_sequence(audio_input, model, encoder_states, latent_dim)
            if predicted_sequence == decode_transcript(transcripts[i], settings.CHARACTER_SET):
                good_prediction += 1

        return good_prediction * 100 / test_size