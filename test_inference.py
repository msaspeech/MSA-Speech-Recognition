import sys
from lib import upload_dataset
from models import train_model, measure_test_accuracy, Seq2SeqModel, Inference
from utils import load_pickle_data
from etc import DRIVE_INSTANCE_PATH, ENCODER_STATES_PATH, TRAINED_MODELS_PATH
from etc import settings
from lib import AudioInput
import numpy as np

settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)

word_level = 0
architecture = 1
latent_dim = 350
epochs = 100

#(train_decoder_input, train_decoder_target), \
#(test_encoder_input, test_decoder_input, test_decoder_target) = upload_dataset(word_level=False)

#print(train_encoder_input.shape, train_decoder_input.shape, train_decoder_target.shape)

architecture_path = TRAINED_MODELS_PATH+"architecture"+str(architecture)+".h5"
inference = Inference(model_path=architecture_path, latent_dim=350)

sample = AudioInput("test.wav", "")
audio_sequence = sample.mfcc.transpose()
audio_sequence = np.array(audio_sequence)

transcript = inference.decode_audio_sequence_character_based(audio_sequence)
print(transcript)

# accuracy = measure_test_accuracy(test_decoder_input, model, encoder_states, latent_dim=512)
