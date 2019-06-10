import sys
from lib import upload_dataset, upload_dataset_partition
from models import Seq2SeqModel
from etc import settings
#from init_directories import init_directories

word_level = int(sys.argv[1])
architecture = int(sys.argv[2])
latent_dim = int(sys.argv[3])
epochs = int(sys.argv[4])

model = Seq2SeqModel(latent_dim=latent_dim, epochs=epochs, model_architecture=architecture, word_level=word_level)
model.test_model()
