from utils import upload_data_after_padding
import pandas as pd


def get_encoder_input_data(train_ratio=0.8):
    train_x, test_x = list(), list()
    data = upload_data_after_padding()
    train_length = int(len(data) * train_ratio)
    for i, audio_sample in enumerate(data):
        if i <= train_length:
            train_x.append(audio_sample.mfcc.transpose())
            train_x = pd.DataFrame.from_records(train_x)
        else:
            test_x.append(audio_sample.mfcc.transpose())
            test_x = pd.DataFrame.from_records(test_x)
    return train_x, test_x


def get_decoder_input_data():

    return None, None


def get_decoder_target_data():

    return None, None


def upload_dataset():
    train_x, test_x = get_input_data()
    train_y, test_y = get_output_data()


