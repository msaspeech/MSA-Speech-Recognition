import re


def get_transcript_key_value(transcript):
    """
    :param transcript:
    :return:
    """
    transcript = transcript.rstrip("\n")
    key_pattern = "(MSA[\d]+) .*"
    value_pattern = "MSA[\d]+ (.*)"
    transcript_wav_file = re.findall(key_pattern, transcript)[0] + ".wav"
    transcript_content = re.findall(value_pattern, transcript)[0]

    return transcript_wav_file, transcript_content


def map_transcripts(self, file_path):
    transcripts_map = dict()
    with open(file_path, "r") as transcripts_file:
        uncleaned_transcripts = transcripts_file.readlines()
        for t in uncleaned_transcripts:
            transcript_key, transcript_value = get_transcript_key_value(t)
            transcripts_map[transcript_key] = transcript_value
    return transcripts_map
