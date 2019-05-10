import re


def special_characters_table():
    special_characters = {}
    special_characters["%"] = "في المئة"
    special_characters["@"] = ""
    special_characters["#"] = ""
    special_characters[";"] = ""
    special_characters["\\"] = ""
    special_characters["B"] = "b"
    special_characters["C"] = "c"
    special_characters["e"] = "E"
    special_characters["G"] = "g"
    special_characters["I"] = "i"
    special_characters["J"] = "j"
    special_characters["L"] = "l"
    special_characters["M"] = "m"
    special_characters["P"] = "p"
    special_characters["Q"] = "q"
    special_characters["R"] = "r"
    special_characters["U"] = "u"
    special_characters["V"] = "v"
    special_characters["W"] = "w"
    special_characters["X"] = "x"
    special_characters["ﻻ"] = ""
    special_characters["ﻹ"] = ""
    special_characters["ﻷ"] = ""
    special_characters["ﻵ"] = ""
    special_characters["٠"] = "0"
    special_characters["١"] = "1"
    special_characters["٢"] = "2"
    special_characters["٣"] = "3"
    special_characters["٤"] = "4"
    special_characters["٦"] = "6"
    special_characters["٩"] = "9"
    special_characters["ﺇ"] = ""

    return special_characters


def _replace_numbers(transcription, num_text_table):
    pattern = "\d+"
    numbers = re.findall(pattern, transcription)
    if numbers :
        for number in numbers:
            corresponding_text = num_text_table[int(number)]
            pattern = str(number)
            transcription = re.sub(pattern, corresponding_text, transcription)

    return transcription


def _replace_special_characters(transcription, special_characters_table):
    for pattern, to_replace_with in special_characters_table.items():
        matches = re.findall(pattern, transcription)
        if matches:
            for match in matches:
                transcription = re.sub(match, to_replace_with, transcription)

    return transcription


def transcript_preprocessing(transcription, num_text_table, special_characters_table):
    #replacing special characters
    transcription = _replace_special_characters(transcription, special_characters_table)
    # Replacing numbers
    transcription = _replace_numbers(transcription, num_text_table)

    return transcription


