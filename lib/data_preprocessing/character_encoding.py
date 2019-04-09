from utils import upload_data_after_padding
import numpy as np


def _get_transcriptions():
    """
    Returns transcripts of the dataset
    :return: List of Strings
    """
    transcripts = []
    data = upload_data_after_padding()
    for audio_sample in data:
        transcripts.append(audio_sample.audio_transcript)
    return transcripts


def _get_distinct_characters(transcripts):
    """
    Gets distincts characters for all dataset
    :param transcripts:
    :return:
    """
    characters_set = set()
    for t in transcripts:
        for c in t:
            if c not in characters_set:
                characters_set.add(c)
    return characters_set


def _convert_to_int(character_set):
    """
    Returns a dict containing the int that corresponds to the char to encode input
    :param character_set: set
    :return: dict
    """
    char_to_int = dict()
    for i, char in enumerate(character_set):
        char_to_int[char] = i
    return char_to_int


def _convert_to_char(character_set):
    """
        Returns a dict containing the char that corresponds to an int to decode target
        :param character_set: set
        :return: dict
        """
    int_to_char = dict()
    for i, char in enumerate(character_set):
        int_to_char[i] = char

    return int_to_char


def _get_longest_sample(transcripts):
    """
    Return the maximum sample length for our dataset
    :param transcripts: List of String
    :return: int
    """
    return max([len(transcript) for transcript in transcripts])


def _generate_input_target_data(transcripts, char_to_int, num_transcripts, max_length, num_distinct_chars):
    """
    Generates two 3D arrays for the decoder input data and target data.
    Fills the 3D arrays for each sample of our dataset
    Return OneHotEncoded Decoder Input data
    Return OneHotEncoded Target data
    :param transcripts: List of Strings
    :param char_to_int: Dict
    :param num_transcripts: int
    :param max_length: int
    :param num_distinct_chars: int
    :return: 3D numpy Array, 3D numpy Array
    """
    #Initializing empty 3D array for decoder input
    decoder_input_data = np.zeros((num_transcripts,
                                   max_length,
                                   num_distinct_chars),
                                   dtype='float32')

    # Initializing empty 3D array for enc/dec target
    target_data = np.zeros((num_transcripts,
                                   max_length,
                                   num_distinct_chars),
                                   dtype='float32')

    #Parsing through transcripts to fill the 3D array
    for i, transcript in enumerate(transcripts):
        for index, character in enumerate(transcript):
            decoder_input_data[i, index, char_to_int[character]] = 1
            if index > 0:
                target_data[i, index-1, char_to_int[character]] = 1

    return decoder_input_data, target_data


def generate_decoder_input_target(transcripts=None):
    """
    Wrapper for the _generate_input_target_data method.
    :return: 3D numpy Array, 3D numpy Array
    """
    if transcripts is None:
        transcripts = _get_transcriptions()
    character_set = _get_distinct_characters(transcripts)
    char_to_int, int_to_char = _convert_to_int(character_set), _convert_to_char(character_set)
    max_sample_length = _get_longest_sample(_get_transcriptions())
    decoder_input, decoder_target = _generate_input_target_data(transcripts,
                                                                char_to_int,
                                                                len(transcripts),
                                                                max_sample_length,
                                                                len(character_set))
    return decoder_input, decoder_target


