from utils import get_files_full_path, load_pickle_data, generate_pickle_file
from models import plot_train_loss_acc


list_files = get_files_full_path("./figures/character")


for file in list_files:
    plot_train_loss_acc(file, word_level=0)
