def get_longest_sample_size(transcripts):
    """
    Return the maximum sample length for our dataset
    :param transcripts: List of String
    :return: int
    """
    return max([len(transcript) for transcript in transcripts])