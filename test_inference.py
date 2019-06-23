import sys
from lib import upload_dataset
from models import train_model, measure_test_accuracy, Seq2SeqModel, Word_Inference, Word_Inference_TEST, Char_Inference

word_level = int(sys.argv[1])
architecture = int(sys.argv[2])
latent_dim = int(sys.argv[3])

from etc import settings
from lib import AudioInput
import numpy as np

#settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)


sample = AudioInput("audio.wav", "")
audio = [sample.mfcc.transpose()]

audio_sequence = np.array(audio, dtype=np.float32)
print(audio_sequence.shape)


char_inference = Char_Inference(word_level=word_level, architecture=architecture, latent_dim=latent_dim)

char_inference.decode_audio_sequence_character_based(audio_sequence)
char_inference.predict_sequence_test(audio_sequence)
