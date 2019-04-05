import glob
import os
import pickle

from lib import AudioInput
from etc import DATA_PATH
from etc import QCRI_TRANSCRIPTS_PATH
from etc import QCRI_CORPUS_PATH
from utils import map_transcripts


def generate_picke_file():
    """
     Loads the mapped audio_transcript data
     Create AudioInput object and assigns its transcript
     Generates pickle File in data directory
    """
    mapped_data = map_transcripts(QCRI_TRANSCRIPTS_PATH)
    with open(DATA_PATH + 'QCRI_data_objects.pkl', 'wb') as f:
        i= 0
        for file in glob.glob(os.path.join(QCRI_CORPUS_PATH, '*.wav')):

            # Retreiving audio transcript for the file
            key = file.split("./data/corpus/audio/")[1]
            audio_transcript = mapped_data[key]

            # Generating audio object
            audio_processing = AudioInput(file)
            audio_processing.set_transcript(audio_transcript)

            # Dumps object in the picke file
            pickle.dump(audio_processing, f, pickle.HIGHEST_PROTOCOL)
