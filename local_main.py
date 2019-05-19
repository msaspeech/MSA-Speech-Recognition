import sys
from lib import upload_dataset
from models import train_model, measure_test_accuracy, Seq2SeqModel
from utils import load_pickle_data
from etc import DRIVE_INSTANCE_PATH
from etc import settings
from init_directories import init_directories

init_directories()
settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)

architecture = 1
word_level = 0
latent_dim = 512
epochs = 100

settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)
upload_dataset(word_level=word_level)
model = Seq2SeqModel(latent_dim=latent_dim, epochs=epochs, model_architecture=1)
model.train_model()


# accuracy = measure_test_accuracy(test_decoder_input, model, encoder_states, latent_dim=512)
