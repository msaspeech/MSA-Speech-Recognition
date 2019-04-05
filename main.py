import glob
import os
import pickle

from etc import QCRI_CORPUS_PATH
from lib import AudioProcessing

for file in glob.glob(os.path.join(QCRI_CORPUS_PATH, '*.wav')):
    audio_processing = AudioProcessing(file)
    data = audio_processing.data
    sample_rate = audio_processing.sample_rate
    mfcc = audio_processing.mfcc
    with open('QCRI_data_objects.pkl', 'wb') as f:
        pickle.dump(audio_processing, f, pickle.HIGHEST_PROTOCOL)