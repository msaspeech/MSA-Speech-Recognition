import sys
from lib import upload_dataset
from models import Seq2SeqModel
from models import train_model, measure_test_accuracy
from utils import load_pickle_data
from etc import DRIVE_INSTANCE_PATH
from etc import settings
from init_directories import init_directories

init_directories()
settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)

word_level = int(sys.argv[1])
architecture = int(sys.argv[2])
latent_dim = int(sys.argv[3])
epochs = int(sys.argv[4])

upload_dataset(word_level=word_level, partitions=16)
model = Seq2SeqModel(latent_dim=latent_dim, epochs=epochs, model_architecture=1, word_level=word_level)
model.train_model()


# accuracy = measure_test_accuracy(test_decoder_input, model, encoder_states, latent_dim=512)
