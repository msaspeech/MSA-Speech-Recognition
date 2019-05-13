import numpy as np
from etc import settings
from utils import generate_pickle_file


def get_attribute_values(dataset, attribute_index):
    attribute_values = []
    for sample in dataset:
        attribute_values.append(sample[:, attribute_index])

    return np.array(attribute_values)


def min_max_normalization(value, index_column, min_attributes, max_attributes):
    normalized = (value - min_attributes[index_column]) / (max_attributes[index_column] - min_attributes[index_column])
    return normalized


def normalize_encoder_input(dataset):

    min_attributes = []
    max_attributes = []
    print("into normalization")
    # Calculating and saving min and max values of dataset
    for attribute_index in range(0, settings.MFCC_FEATURES_LENGTH):

            attribute_values = get_attribute_values(dataset, attribute_index)
            min_attributes.append(np.min(attribute_values))
            max_attributes.append(np.max(attribute_values))

    print(min_attributes)
    print(max_attributes)
    generate_pickle_file(min_attributes, settings.ENCODER_INPUT_MIN_VALUES_PATH)
    generate_pickle_file(max_attributes, settings.ENCODER_INPUT_MAX_VALUES_PATH)

    print("generating new dataset")
    for i, encoder_input in enumerate(dataset):
        normalized_encoder_input = []
        for line in encoder_input:
            normalized_line = []
            for index_column, value in enumerate(line):
                normalized_line.append(min_max_normalization(value, index_column, min_attributes, max_attributes))
            normalized_encoder_input.append(normalized_line)
        dataset[i] = normalized_encoder_input

    generate_pickle_file(dataset, settings.NORMALIZED_ENCODER_INPUT_PATH)
    return dataset
