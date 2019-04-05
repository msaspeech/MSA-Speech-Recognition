import re
from etc import QCRI_TRANSCRIPTS_PATH

def get_transcript_key_value(transcript):
    """
    Retreives file name and its transcript for each line of the transcriptions file
    :param transcript: String
    :return: String, String
    """
    transcript = transcript.rstrip("\n")
    key_pattern = "(MSA[\d]+) .*"
    value_pattern = "MSA[\d]+ (.*)"
    transcript_wav_file = re.findall(key_pattern, transcript)[0] + ".wav"
    transcript_content = re.findall(value_pattern, transcript)[0]

    return transcript_wav_file, transcript_content


def map_transcripts(file_path):
    """
    Generates a dict where the key is the audio file name and the value is its transcript
    :param file_path: String
    :return: Dict
    """
    transcripts_map = dict()
    with open(file_path, "r") as transcripts_file:
        uncleaned_transcripts = transcripts_file.readlines()
        for t in uncleaned_transcripts:
            transcript_key, transcript_value = get_transcript_key_value(t)
            transcripts_map[transcript_key] = transcript_value
    return transcripts_map
