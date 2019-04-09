from utils import upload_original_data, upload_data_after_padding, get_longest_sample_size, get_character_set
from . import generate_decoder_input_target
import numpy as np


def _get_train_test_data(train_ratio=0.8, padding=True):
    """
    Splits dataset into train and test according to a ratio
    :param train_ratio: float
    :param padding: Boolean
    :return: List of InputAudio, List of InputAudio
    """
    if padding is False:
        data = upload_original_data()
    else:
        data = upload_data_after_padding()

    train_length = int(len(data) * train_ratio)
    train_data = []
    test_data = []
    for i, audio_sample in enumerate(data):
        if i <= train_length:
            train_data.append(audio_sample)
        else:
            test_data.append(audio_sample)

    return train_data, test_data


def _get_audio_transcripts(data):
    """
    Returns a list of audio mfcc dta and list of transcripts
    :param data: List of Audio Input
    :return: List of ndArray, List of Strings
    """
    audio_samples = []
    transcripts = []

    for sample in data:
        audio_samples.append(sample.mfcc.transpose())
        transcripts.append(sample.audio_transcript)

    return audio_samples, transcripts


def _get_encoder_input_data(audio_data):
    """
    Concatenate list of numpy 2dArray into a 3D numpy Array
    :param audio_data:
    :return: numpy 3dArray
    """
    return np.array(audio_data)


def upload_dataset(train_ratio=0.8, padding=True):
    """
    Generate :
    train ==> encoder inputs, decoder inputs, decoder target
    test ==>  encoder inputs, decoder inputs, decoder target
    :return: Tuple, Tuple
    """

    # Upload train and test data, the train ration is 0.8 and can be modified through ration param
    train_data, test_data = _get_train_test_data(train_ratio=train_ratio, padding=padding)

    # get mfcc and text transcripts for train and test
    train_audio, train_transcripts = _get_audio_transcripts(train_data)
    test_audio, test_transcripts = _get_audio_transcripts(test_data)

    # get max transcript size and character_set
    all_transcripts = train_transcripts + test_transcripts
    transcript_max_length = get_longest_sample_size(all_transcripts)
    character_set = get_character_set(all_transcripts)

    # generate 3D numpy arrays for train encoder inputs and test encoder inputs
    train_encoder_input = _get_encoder_input_data(train_audio)
    test_encoder_input = _get_encoder_input_data(test_audio)

    # generate 3D numpy arrays for train and test decoder input and decoder target
    train_decoder_input, train_decoder_target = generate_decoder_input_target(character_set=character_set,
                                                                              length= transcript_max_length,
                                                                              transcripts=train_transcripts)

    test_decoder_input, test_decoder_target = generate_decoder_input_target(character_set=character_set,
                                                                            length=transcript_max_length,
                                                                            transcripts=test_transcripts)

    return (train_encoder_input, train_decoder_input, train_decoder_target), \
           (test_encoder_input, test_decoder_input, test_decoder_target)


