from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D


def get_cnn_model(input_shape):
    """
    Creates a Sequential model with Conv and MaxPooling layers
    Prints summary of model
    :param input_shape: tuple
    :return: Sequential
    """

    model = Sequential()
    model.add(Conv1D(16, 16, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 16, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 16, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 4, activation='relu'))
    model.add(MaxPooling1D(2))
    return model
