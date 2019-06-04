import os
from utils import file_exists
from etc import settings


def init_directories():
    list_paths = [settings.DATA_PATH, settings.DRIVE_PATH, settings.NORMALIZATION_PATH, settings.DATASET_SPLIT_PATH, settings.DATASET_SPLIT_TRAIN_PATH,
                  settings.DATASET_SPLIT_TEST_PATH, settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH, settings.TRANSCRIPTS_ENCODING_SPLIT_TEST_PATH,
                  settings.AUDIO_SPLIT_TRAIN_PATH, settings.AUDIO_SPLIT_TEST_PATH, settings.TRAINED_MODELS_PATH, settings.TRAIN_HISTORY]


    for path in list_paths:
        if not file_exists(path):
            os.mkdir(path)

    history_models_path = settings.TRAIN_HISTORY + "architecture"

    for i in range(1, 8):
        directory_path = history_models_path + str(i) + "/"
        if not file_exists(directory_path):
            os.mkdir(directory_path)

