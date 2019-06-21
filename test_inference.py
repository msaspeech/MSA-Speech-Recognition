import sys
from lib import upload_dataset
from models import train_model, measure_test_accuracy, Seq2SeqModel, Word_Inference, Word_Inference_TEST, Char_Inference

word_level = 0
architecture = 1
latent_dim = 500

from etc import settings
from lib import AudioInput
import numpy as np

#settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)



model_name = "architecture" + str(architecture)
if word_level:
    model_path = settings.TRAINED_MODELS_PATH + model_name + "/" + model_name + "word.h5"
else:
    model_path = settings.TRAINED_MODELS_PATH + model_name + "/" + model_name + "char.h5"

print(model_path)

sample = AudioInput("audio.wav", "")
audio = [sample.mfcc.transpose()]

audio_sequence = np.array(audio, dtype=np.float32)
print(audio_sequence.shape)


char_inference = Char_Inference(model_path=model_path, latent_dim=latent_dim)

char_inference.decode_audio_sequence_character_based(audio_sequence)
char_inference.predict_sequence_test(audio_sequence)
