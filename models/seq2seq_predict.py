import numpy as np
from utils import convert_to_int, convert_to_char


def decode_audio_sequence(audio_sequence, encoder_model, decoder_model, character_set):
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
