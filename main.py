import sys
from lib import upload_dataset, upload_dataset_partition, upload_dataset_2
from models import Seq2SeqModel
from models import train_model, measure_test_accuracy
from utils import load_pickle_data
from etc import DRIVE_INSTANCE_PATH
from etc import settings
from init_directories import init_directories

init_directories()
#settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)

word_level = int(sys.argv[1])
architecture = int(sys.argv[2])
latent_dim = int(sys.argv[3])
epochs = int(sys.argv[4])
partitions = int(sys.argv[5])

upload_dataset_partition(word_level=word_level, partitions=partitions)
model = Seq2SeqModel(latent_dim=latent_dim, epochs=epochs, model_architecture=architecture, word_level=word_level)
print("yeah here")
model.train_model()

#(train_encoder_input, train_decoder_input, train_decoder_target), \
#(test_encoder_input, test_decoder_input, test_decoder_target) = upload_dataset_2()


#model.train_model(train_encoder_input, train_decoder_input, train_decoder_target)


# accuracy = measure_test_accuracy(test_decoder_input, model, encoder_states, latent_dim=512)
