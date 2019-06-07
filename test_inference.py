import sys
from lib import upload_dataset
from models import train_model, measure_test_accuracy, Seq2SeqModel, Word_Inference

from etc import settings
from lib import AudioInput
import numpy as np

#settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)

word_level = 1
architecture = 1
latent_dim = 400


model_name = "architecture" + str(architecture)
if word_level:
    model_path = settings.TRAINED_MODELS_PATH + model_name + "/" + model_name + "word.h5"
else:
    model_path = settings.TRAINED_MODELS_PATH + model_name + "/" + model_name + "char.h5"

sample = AudioInput("test.wav", "")
audio = [sample.mfcc.transpose()]

audio_sequence = np.array(audio, dtype=np.float32)
print(audio_sequence.shape)

word_inference = Word_Inference(model_path=model_path, latent_dim=latent_dim)

transcript = word_inference.decode_audio_sequence(audio_sequence)
print(transcript)

# accuracy = measure_test_accuracy(test_decoder_input, model, encoder_states, latent_dim=512)
