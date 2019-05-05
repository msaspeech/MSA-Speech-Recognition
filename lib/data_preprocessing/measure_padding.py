import numpy as np


def _sum_timelapses(timelapses):
    """
    Returns sum of time lapses
    :param timelapses: List
    :return: integer
    """
    sum = 0
    for elt in timelapses:
        sum += elt
    return sum


def _retrieve_timelapse(audio_data):
    """
    Returns a list of timelapses of each AudioInput object
    :param audio_data: List of AudioInput objects
    :return: List
    """
    timelapses = []
    for audio_sample in audio_data :
        timelapses.append(audio_sample.mfcc.shape[1])
    return timelapses


def calculate_padding(audio_data, measure_unit):
    """
    Gets as parameter measure unit (average/3rd quantile)
    Returns corresponding width pad
    :param audio_data: List
    :param measure_unit: String
    :return: Integer
    """
    timelapses = _retrieve_timelapse(audio_data)
    if measure_unit is "avg":
        timelapses_sum = _sum_timelapses(timelapses)
        return int(timelapses_sum/len(timelapses))
    elif measure_unit is "q3":
        return int(np.quantile(timelapses, .75))

