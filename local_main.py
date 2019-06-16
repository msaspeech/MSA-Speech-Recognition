import sys
from lib import upload_dataset, upload_dataset_partition, upload_dataset_2
from models import train_model, measure_test_accuracy, Seq2SeqModel
from utils import load_pickle_data
from etc import DRIVE_INSTANCE_PATH
from etc import settings
from init_directories import init_directories

init_directories()
settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)

architecture = 1
word_level = 0
latent_dim = 350
epochs = 500

upload_dataset_partition(word_level=word_level, partitions=2)
model = Seq2SeqModel(latent_dim=latent_dim, epochs=epochs, model_architecture=architecture, word_level=word_level)
model.train_model()



#(train_encoder_input, train_decoder_input, train_decoder_target), \
#(test_encoder_input, test_decoder_input, test_decoder_target) = upload_dataset_2()

#print(train_encoder_input.shape, train_decoder_input.shape, train_decoder_target.shape)

#model.train_model(train_encoder_input, train_decoder_input, train_decoder_target)

# accuracy = measure_test_accuracy(test_decoder_input, model, encoder_states, latent_dim=512)
