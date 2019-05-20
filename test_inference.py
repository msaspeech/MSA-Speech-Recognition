import sys
from lib import upload_dataset
from models import train_model, measure_test_accuracy, Seq2SeqModel, Inference
from utils import load_pickle_data
from etc import DRIVE_INSTANCE_PATH, ENCODER_STATES_PATH, TRAINED_MODELS_PATH
from etc import settings


settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)

word_level = 0
architecture = 1
latent_dim = 350
epochs = 100

#(train_decoder_input, train_decoder_target), \
#(test_encoder_input, test_decoder_input, test_decoder_target) = upload_dataset(word_level=False)

#print(train_encoder_input.shape, train_decoder_input.shape, train_decoder_target.shape)

architecture_path = TRAINED_MODELS_PATH+"architecture"+str(architecture)+".h5"
encoder_states_path = ENCODER_STATES_PATH+"architecture"+str(architecture)+".pkl"
inference = Inference(model_path=architecture_path, encoder_states_path=encoder_states_path, latent_dim=350)


# accuracy = measure_test_accuracy(test_decoder_input, model, encoder_states, latent_dim=512)
