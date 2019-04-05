import numpy as np
from utils import upload_original_data, generate_pickle_file_padding
from lib import calculate_padding


def get_resized_mfcc(mfcc, pad_width):
    """
    Reshapes the mfcc matrix to a unified size
    :param mfcc: 2D array containing mfcc spectrogram
    :param pad_width: integer
    :return: 2D array
    """
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc


def generate_fixed_size_data():
    """
    Updates mfcc attribute for each AudioInput Object
    Saves the AudioInput objects after padding in a pickle file
    :return:
    """
    updated_data = []
    original_sized_data = upload_original_data()
    max_pad_len = calculate_padding(original_sized_data, "q3")
    for audio_sample in original_sized_data:
        pad_width = max_pad_len - audio_sample.mfcc.shape[1]
        if pad_width > 0 :
            audio_sample.mfcc = get_resized_mfcc(audio_sample.mfcc, pad_width)
            updated_data.append(audio_sample)
    generate_pickle_file_padding(updated_data)

