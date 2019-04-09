def get_longest_sample_size(transcripts):
    """
    Return the maximum sample length for our dataset
    :param transcripts: List of String
    :return: int
    """
    return max([len(transcript) for transcript in transcripts])


def convert_to_int(character_set):
    """
    Returns a dict containing the int that corresponds to the char to encode input
    :param character_set: set
    :return: dict
    """
    char_to_int = dict()
    for i, char in enumerate(character_set):
        char_to_int[char] = i
    return char_to_int


def convert_to_char(character_set):
    """
        Returns a dict containing the char that corresponds to an int to decode target
        :param character_set: set
        :return: dict
        """
    int_to_char = dict()
    for i, char in enumerate(character_set):
        int_to_char[i] = char

    return int_to_char
