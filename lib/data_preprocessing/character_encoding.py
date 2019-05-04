from utils.manage_transcripts import convert_to_int, get_longest_sample_size
import numpy as np


def _get_transcriptions(audioInput_data):
    """
    Returns transcripts of the dataset
    :return: List of Strings
    """
    transcripts = []
    for audio_sample in audioInput_data:
        transcripts.append(audio_sample.audio_transcript)
    return transcripts


def one_hot_encode_transcript(transcript, char_to_int, num_distinct_chars):
    """
    One hot encodes a transcript to input_transcript and target_transcript which is the input_transcript at t+1
    :param transcript: String
    :param char_to_int: Dict
    :param num_distinct_chars: Int
    :return: Numpy 2dArray, Numpy 2dArray
    """
    input_transcript = np.zeros((len(transcript),
                                num_distinct_chars),
                                dtype='float32')

    target_transcript = np.zeros((len(transcript),
                                 num_distinct_chars),
                                 dtype='float32')
    for index, character in enumerate(transcript):
        input_transcript[index, char_to_int[character]] = 1
        if index > 0:
            input_transcript[index-1, char_to_int[character]] = 1

    return input_transcript, target_transcript


def _generate_input_target_data(transcripts, char_to_int, num_distinct_chars):
    """
    Generates two 3D arrays for the decoder input data and target data.
    Fills the 3D arrays with each sample of our dataset
    Returns OneHotEncoded Decoder Input data
    Returns OneHotEncoded Target data
    :param transcripts: List of Strings
    :param char_to_int: Dict
    :param num_distinct_chars: int
    :return: 3D numpy Array, 3D numpy Array
    """
    encoded_decoder_inputs = []
    encoded_decoder_targets = []

    for transcript in transcripts:
        encoded_input, encoded_target = one_hot_encode_transcript(transcript=transcript,
                                                                  char_to_int=char_to_int,
                                                                  num_distinct_chars=num_distinct_chars)
        encoded_decoder_inputs.append(encoded_input)
        encoded_decoder_targets.append(encoded_target)

    decoder_input_data = np.array(encoded_decoder_inputs)
    decoder_target_data = np.array(encoded_decoder_targets)

    return decoder_input_data, decoder_target_data


def _generate_fixed_size_input_target_data(transcripts, char_to_int, num_transcripts, max_length, num_distinct_chars):
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
    # Initializing empty 3D array for decoder input
    decoder_input_data = np.zeros((num_transcripts,
                                   max_length,
                                   num_distinct_chars), dtype='float32')

    # Initializing empty 3D array for enc/dec target
    decoder_target_data = np.zeros((num_transcripts,
                                   max_length,
                                   num_distinct_chars), dtype='float32')

    # Parsing through transcripts to fill the 3D array
    for i, transcript in enumerate(transcripts):
        for index, character in enumerate(transcript):
            decoder_input_data[i, index, char_to_int[character]] = 1
            if index > 0:
                decoder_target_data[i, index-1, char_to_int[character]] = 1

    return decoder_input_data, decoder_target_data


def generate_decoder_input_target(audioInput_data, character_set, transcripts=None):
    """
    Wrapper for the _generate_input_target_data method.
    :return: 3D numpy Array, 3D numpy Array
    """
    if transcripts is None:
        transcripts = _get_transcriptions(audioInput_data)
    char_to_int = convert_to_int(character_set)
    max_transcript_length = get_longest_sample_size(transcripts)
    decoder_input, decoder_target = _generate_fixed_size_input_target_data(transcripts=transcripts,
                                                                           char_to_int=char_to_int,
                                                                           num_transcripts=len(transcripts),
                                                                           max_length=max_transcript_length,
                                                                           num_distinct_chars=len(character_set))
    #decoder_input, decoder_target = _generate_input_target_data(transcripts,
    #                                                           char_to_int,
    #                                                            len(character_set))
    return decoder_input, decoder_target


