import os
from etc import RAW_DATASET_AUDIO_PATH, RAW_DATASET_TRANSCRIPTIONS, GENERATED_DATA_PATH
from . import generate_transcriptions_file, split_audio


def get_files(directory):
    list_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            list_files.append(directory + filename)

    return list_files


def get_audio_transcripts_pairs():
    audio_transcripts_descriptions = []
    audio_files = get_files(RAW_DATASET_AUDIO_PATH)
    audio_descriptions = get_files(RAW_DATASET_TRANSCRIPTIONS)
    for i, _ in enumerate(audio_files):
        audio_transcripts_descriptions.append((audio_files[i], audio_descriptions[i]))

    return audio_transcripts_descriptions


def generate_dataset():
    audio_descriptions_pairs = get_audio_transcripts_pairs()
    for i, (audio_entry, transcript_desc) in enumerate(audio_descriptions_pairs):

        transcript_path = GENERATED_DATA_PATH+"audio"+str(i)+"/audio"+str(i)+"_transcript.txt"
        generate_transcriptions_file(input_path=transcript_desc, output_path=transcript_path)

        audio_path = GENERATED_DATA_PATH+"wav/"
        split_audio(audio_entry=audio_entry, audio_output_path=audio_path)


