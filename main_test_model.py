import sys
from lib import upload_dataset, upload_dataset_partition
from models import Seq2SeqModel
from etc import settings
from utils import load_pickle_data

word_level = int(sys.argv[1])
architecture = int(sys.argv[2])
latent_dim = int(sys.argv[3])
epochs = int(sys.argv[4])


general_info = load_pickle_data(settings.DATASET_WORD_INFORMATION_PATH)
settings.MFCC_FEATURES_LENGTH = general_info[0]
settings.TOTAL_SAMPLES_NUMBER = general_info[1]
settings.WORD_SET = general_info[2]
settings.LONGEST_WORD_LENGTH = general_info[3]
settings.CHARACTER_SET = general_info[4]
settings.WORD_TARGET_LENGTH = general_info[5]


model = Seq2SeqModel(latent_dim=latent_dim, epochs=epochs, model_architecture=architecture, word_level=word_level)
model.test_model()
