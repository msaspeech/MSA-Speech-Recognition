import numpy as np
from utils import convert_to_int, convert_to_char, decode_transcript
from tensorflow.python.keras import models
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input
from etc import settings


class Inference():
    def __init__(self, model_path, encoder_states, latent_dim):
        self.model = models.load_model(model_path)
        self.encoder_states = encoder_states
        self.latent_dim = latent_dim
        self.encoder_model, self.decoder_model = self._get_encoder_decoder_model_base_line()
        self.character_set = settings.CHARACTER_SET

    def _get_encoder_decoder_model_base_line(self):
        # Getting layers after training (updated weights)
        encoder_inputs = self.model.get_layer("encoder_inputs")
        decoder_inputs = self.model.get_layer("decoder_inputs")
        decoder_lstm2_layer = self.model.get_layer("decoder_lstm2_layer")
        decoder_lstm1_layer = self.model.get_layer("decoder_lstm1_layer")
        decoder_dense = self.model.get_layer("decoder_dense")

        # Creating encoder model
        encoder_model = Model(encoder_inputs, self.encoder_states)

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
        char_to_int = convert_to_int(character_set)
        int_to_char = convert_to_char(character_set)

        states_value = encoder_model.predict(audio_sequence)

        num_decoder_tokens = len(char_to_int)
        target_sequence = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first character of target sequence with the start character.
        target_sequence[0, 0, char_to_int['\t']] = 1.

        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_sequence] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = char_to_int[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == "\n":
                # End of transcription
                stop_condition = True
            else:
                # updating target sequence vector
                target_sequence[0, 0, sampled_token_index] = 1

            states_values = [h, c]

        return decoded_sentence