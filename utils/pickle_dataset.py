from dataset_generation.audio_transcript_map import map_audio_transcripts
from utils import generate_pickle_file
from lib import AudioInput
from lib import get_fixed_size_data
from etc import PICKLE_PAD_FILE_PATH


def generate_pickle_dataset():
    mapped_audio = map_audio_transcripts()
    audioInput = []
    for path, transcription in mapped_audio.items():
        print(path+"====>"+transcription)
        audioInput.append(AudioInput(path=path, transcript=transcription))

    updated_data = get_fixed_size_data(audioInput_data=audioInput)

    generate_pickle_file(updated_data, PICKLE_PAD_FILE_PATH)

