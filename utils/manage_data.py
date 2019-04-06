import glob
import os
import pickle

from lib import AudioInput
from etc import DATA_PATH
from etc import QCRI_TRANSCRIPTS_PATH
from etc import QCRI_CORPUS_PATH
from etc import PICKLE_FILE_PATH
from utils import map_transcripts


def generate_pickle_file():
    """
     Loads the mapped audio_transcript data
     Create AudioInput object and assigns its transcript
     Generates pickle File in data directory
    """
    mapped_data = map_transcripts(QCRI_TRANSCRIPTS_PATH)
    audio_data = []
    with open(DATA_PATH + 'QCRI_data_objects.pkl', 'wb') as f:
        for file in glob.glob(os.path.join(QCRI_CORPUS_PATH, '*.wav')):

            # Retreiving audio transcript for the file
            key = file.split("./data/corpus/audio/")[1]
            audio_transcript = mapped_data[key]

            # Generating audio object
            audio_informations = AudioInput(file)
            audio_informations.set_transcript(audio_transcript)
            audio_data.append(audio_informations)
            # Dumps object in the picke file
        pickle.dump(audio_data, f, pickle.HIGHEST_PROTOCOL)


def generate_pickle_file_padding(updated_data):
    """
    Generates pickle file in directory with data after padding
    :param updated_data: List of AudioInput objects
    """
    with open(DATA_PATH + 'QCRI_data_objects_pad.pkl', 'wb') as f:
        pickle.dump(updated_data, f, pickle.HIGHEST_PROTOCOL)


def upload_original_data():
    """
    Uploads AudioInput data from pickle file
    :return:
    """
    with open(PICKLE_FILE_PATH, "rb") as f:
       data = pickle.load(f)
    return data


def upload_data_after_padding():
    """
      Uploads AudioInput data after padding from pickle file
      :return:
      """
    with open(PICKLE_FILE_PATH, "rb") as f:
        data = pickle.load(f)
    return data
