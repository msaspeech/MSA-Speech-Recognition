import os
from etc import RAW_DATASET_AUDIO_PATH, RAW_DATASET_TRANSCRIPTIONS, GENERATED_DATA_PATH
from . import generate_transcriptions_file, get_transcriptions, split_audio


def get_files(directory):
    list_files = []
    for root, dirs, files in os.walk(directory):
        for filename in sorted(files):
            list_files.append(directory + filename)

    return list_files


def get_audio_transcripts_pairs(audio_files_path, transcription_files_path):

    audio_transcripts_descriptions = []

    audio_files = get_files(audio_files_path)
    transcription_files = get_files(transcription_files_path)

    for i, _ in enumerate(audio_files):
        audio_transcripts_descriptions.append((audio_files[i], transcription_files[i]))

    return audio_transcripts_descriptions


def generate_dataset():
    audio_descriptions_pairs = get_audio_transcripts_pairs(RAW_DATASET_AUDIO_PATH, RAW_DATASET_TRANSCRIPTIONS)

    for i, (audio_entry, transcript_desc_path) in enumerate(audio_descriptions_pairs):
        transcriptions_directory = GENERATED_DATA_PATH+"transcriptions/"
        if not os.path.exists(transcriptions_directory):
            os.mkdir(transcriptions_directory)

        transcript_path = transcriptions_directory+"audio"+str(i)+"_transcript.txt"
        transcriptions_description = get_transcriptions(transcript_desc_path)
        generate_transcriptions_file(transcriptions_desc=transcriptions_description,
                                     output_path=transcript_path)

        audio_directory = GENERATED_DATA_PATH + "wav/audio" + str(i) + "/"
        if not os.path.exists(audio_directory):
            os.mkdir(audio_directory)

        split_audio(audio_entry=audio_entry,
                    transcriptions_desc=transcriptions_description,
                    audio_output_dir=audio_directory)


