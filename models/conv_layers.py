from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D


def generate_cnn_layers(input_shape):
    """
    Creates a Sequential model with Conv and MaxPooling layers
    Prints summary of model
    :param input_shape: tuple
    :return: Sequential
    """
    model = Sequential()
    model.add(Conv1D(64, 15, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(4))
    model.add(Conv1D(128, 15, activation='relu'))
    model.add(MaxPooling1D(4))
    print(model.summary())
    return model


def return_cnn_layers(input_shape):
    """
    Returns list with Conv and MaxPooling layers to be used in another model
    :param input_shape: tuple
    :return: list
    """
    layers_list = []
    layers_list.append(Conv1D(64, 15, activation='relu', input_shape=input_shape))
    layers_list.append(MaxPooling1D(4))
    layers_list.append(Conv1D(128, 15, activation='relu'))
    layers_list.append(MaxPooling1D(4))
    return layers_list

