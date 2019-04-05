import librosa
import matplotlib.pyplot as plt
import librosa.display


# TODO : Add doc to this class
class AudioInput:
    def __init__(self, path):
        self.data, self.sample_rate = librosa.load(path)
        self.mfcc = self.extract_mfcc(self.data, self.sample_rate)
        self.audio_lengh = self.get_audio_length()
        self.audio_path = path

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

    def get_audio_length(self):
        """
        Extract audio lenght in seconds
        :param data: numpy.ndarray
                audio time series
        :param sample_rate: int
                sampling rate of data
        :return: float
                audio length in seconds
        """
        return len(self.data) / self.sample_rate

    # TODO : Add the possibility to save a spectrogram
    def show_audio_spectrogram(self):
        """
        Plot spectrogram
        :param mfcc: numpy.ndarray
                MFCC sequence
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(self.mfcc, x_axis='time')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
