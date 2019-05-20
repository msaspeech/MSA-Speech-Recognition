from utils import  get_character_set, get_distinct_words, empty_directory
from . import generate_decoder_input_target, normalize_encoder_input
import numpy as np
from etc import settings
from utils import load_pickle_data, file_exists, generate_pickle_file
from etc import PICKLE_PAD_FILE_PATH, PICKLE_FILE_PATH, NORMALIZED_ENCODER_INPUT_PATH
import gc


def _get_train_test_data(train_ratio=0.8, padding=False):
    """
    Splits dataset into train and test according to a ratio
    :param train_ratio: float
    :param padding: Boolean
    :return: List of InputAudio, List of InputAudio
    """
    if padding is False:
        data = load_pickle_data(PICKLE_FILE_PATH)
    else:
        data = load_pickle_data(PICKLE_PAD_FILE_PATH)

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
        if 130 <= sample.mfcc.shape[1] <= 1000:
            audio_samples.append(sample.mfcc.transpose())
            transcript = "\t" + sample.audio_transcript + "\n"
            transcripts.append(transcript)

    return audio_samples, transcripts


def _get_audio_transcripts_word_level(data):
    audio_samples = []
    transcripts = []

    for sample in data:
        if 130 <= sample.mfcc.shape[1] <= 1000:
            audio_samples.append(sample.mfcc.transpose())
            transcript = "SOS_ " + sample.audio_transcript + " _EOS"
            transcripts.append(transcript)

    return audio_samples, transcripts


def _generate_spllited_encoder_input_data(audio_data, partitions=8, test=False):
    audio_sets = []
    limits = []
    for i in range(1, partitions + 1):
        limits.append(int(len(audio_data) * i / partitions))

    audio_sets.append(audio_data[0: limits[0]])
    for i in range(1, partitions):
        audio_sets.append(audio_data[limits[i - 1]:limits[i]])

    # Delete original dataset
    audio_data = []
    gc.collect()
    for index, audio_set in enumerate(audio_sets):
        audio_set = np.array(audio_set)
        if not test:
            path = settings.AUDIO_SPLIT_TRAIN_PATH + "audio_set" + str(index) + ".pkl"
        else:
            path = settings.AUDIO_SPLIT_TEST_PATH + "audio_set" + str(index) + ".pkl"

        generate_pickle_file(audio_set, path)


def _get_encoder_input_data(audio_data):
    """
    Concatenate list of numpy 2dArray into a 3D numpy Array
    :param audio_data:
    :return: numpy 3dArray
    """
    return np.array(audio_data)


def upload_dataset(train_ratio=0.8, padding=False, word_level=False, partitions=8):
    """
    Generate :
    train ==> encoder inputs, decoder inputs, decoder target
    test ==>  encoder inputs, decoder inputs, decoder target
    :return: Tuple, Tuple
    """
    if empty_directory(settings.AUDIO_SPLIT_TRAIN_PATH):

        # Upload train and test data, the train ration is 0.8 and can be modified through ration param
        train_data, test_data = _get_train_test_data(train_ratio=0.75, padding=padding)
        settings.TOTAL_SAMPLES_NUMBER = len(train_data)
        print(settings.TOTAL_SAMPLES_NUMBER)
        # get mfcc and text transcripts for train and test
        if word_level:
            train_audio, train_transcripts = _get_audio_transcripts_word_level(train_data)
            # train_audio, train_transcripts = print_suspicious_characters(train_data)
            test_audio, test_transcripts = _get_audio_transcripts_word_level(test_data)

        else:
            train_audio, train_transcripts = _get_audio_transcripts(train_data)
            # train_audio, train_transcripts = print_suspicious_characters(train_data)
            test_audio, test_transcripts = _get_audio_transcripts(test_data)

        # Saving mfcc features and length for global use
        settings.MFCC_FEATURES_LENGTH = train_audio[0].shape[1]
        general_info = [settings.MFCC_FEATURES_LENGTH, settings.TOTAL_SAMPLES_NUMBER]
        # get max transcript size and character_set
        all_transcripts = train_transcripts + test_transcripts
        # transcript_max_length = get_longest_sample_size(all_transcripts)

        _generate_spllited_encoder_input_data(train_audio,partitions=partitions)
        # train_encoder_input = _get_encoder_input_data(audio_data=train_audio)
        _generate_spllited_encoder_input_data(test_audio, test=True,partitions=partitions)

        if word_level:
            distinct_words = get_distinct_words(transcripts=all_transcripts)
            #generate_pickle_file(distinct_words, file_path=settings.DISTINCT_WORDS_PATH)
            general_info.append(distinct_words)
            settings.WORD_SET = distinct_words
        else:
            distinct_characters = get_character_set(transcripts=all_transcripts)
            #generate_pickle_file(distinct_characters, file_path=settings.DISTINCT_CHARACTERS_PATH)
            general_info.append(distinct_characters)
            settings.CHARACTER_SET = distinct_characters

        generate_pickle_file(general_info, settings.DATASET_INFORMATION_PATH)
        generate_decoder_input_target(transcripts=train_transcripts,
                                      word_level=word_level,
                                      partitions=partitions,
                                      fixed_size=False)

        generate_decoder_input_target(transcripts=test_transcripts,
                                      word_level=word_level,
                                      partitions=partitions,
                                      fixed_size=False,
                                      test=True)

    else:
        general_info = load_pickle_data(settings.DATASET_INFORMATION_PATH)
        settings.MFCC_FEATURES_LENGTH = general_info[0]
        settings.TOTAL_SAMPLES_NUMBER = general_info[1]
        if word_level:
            #distinct_words = load_pickle_data(settings.DISTINCT_WORDS_PATH)
            settings.WORD_SET = general_info[2]
            print(settings.WORD_SET)
        else:
            #distinct_characters = load_pickle_data(settings.DISTINCT_CHARACTERS_PATH)
            settings.CHARACTER_SET = general_info[2]
            print(settings.CHARACTER_SET)


    # return (train_decoder_input, train_decoder_target), \
    #       (test_encoder_input, test_decoder_input, test_decoder_target)
