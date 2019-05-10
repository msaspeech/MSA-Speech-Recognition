import re



def _replace_numbers(transcription, num_text_table):
    pattern = "\d+"
    numbers = re.findall(pattern, transcription)
    if numbers :
        for number in numbers:
            corresponding_text = num_text_table[int(number)]
            pattern = str(number)
            transcription = re.sub(pattern, corresponding_text, transcription)

    return transcription


def transcript_preprocessing(transcription, num_text_table):
    # Replacing numbers
    transcription = _replace_numbers(transcription, num_text_table)

    return transcription


