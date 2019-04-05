import librosa
import matplotlib.pyplot as plt
import librosa.display


# TODO : Add doc to this class
class AudioProcessing:
    def __init__(self, filename):
        self.data, self.sample_rate = librosa.load(filename)
        self.mfcc = self.extract_mfcc(self.data, self.sample_rate)

    def extract_mfcc(self, data, sample_rate):
        """
        Extract MFCC sequence from audio.
        :param data: numpy.ndarray
                audio time series
        :param sample_rate: int
                sampling rate of data
        :return: numpy.ndarray
                MFCC sequence
        """
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        return mfcc

    def get_audio_length(self, data, sample_rate):
        """
        Extract audio lenght in seconds
        :param data: numpy.ndarray
                audio time series
        :param sample_rate: int
                sampling rate of data
        :return: float
                audio length in seconds
        """
        return len(data) / sample_rate

    # TODO : Add the possibility to save a spectrogram
    def show_audio_spectrogram(self, mfcc):
        """
        Plot spectrogram
        :param mfcc: numpy.ndarray
                MFCC sequence
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
