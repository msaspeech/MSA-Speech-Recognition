from tensorflow.python.keras.callbacks import Callback, History
from . import plot_train_loss_acc
from etc import ENCODER_STATES_PATH, settings
from utils import generate_pickle_file, file_exists, load_pickle_data, generate_json_file, load_json_data


class ModelSaver(Callback):

    def __init__(self, model_name, model_path, encoder_states, drive_instance, word_level):
        super().__init__()

        self.model_name = model_name
        self.model_path = model_path
        self.word_level = word_level
        self.encoder_states = encoder_states
        self.drive_instance = drive_instance

    def on_epoch_end(self, epoch, logs=None):
        # Saving training history

        print("LOGS" + str(logs))
        if self.word_level:
            hist_path = settings.TRAIN_HISTORY + self.model_name + "word.pkl"
        else:
            hist_path = settings.TRAIN_HISTORY + self.model_name + "char.pkl"

        if file_exists(hist_path):
            acc_loss_history = load_pickle_data(hist_path)
        else:
            acc_loss_history = dict()
            acc_loss_history["accuracy"] = []
            acc_loss_history["loss"] = []

        acc_loss_history["accuracy"].append(logs["acc"])
        acc_loss_history["loss"].append(logs["loss"])

        generate_pickle_file(acc_loss_history, hist_path)
        plot_train_loss_acc(hist_path)

        self.model.save(self.model_path)
        model_title = self.model_name

        # Saving encoder states
        #path = ENCODER_STATES_PATH + model_title + ".pkl"
        #encoder_states = [self.encoder_states]
        #generate_pickle_file(encoder_states, path)

        # Saving model
        parent_directory_id = '0B5fJkPjHLj3Jdkw5ZnFiY0lZV1U'
        file_list = self.drive_instance.ListFile({'q': "\'"+parent_directory_id+"\'"+" in parents  and trashed=false"}).GetList()
        try:
            for file1 in file_list:
                if file1['title'] == self.model_path:
                    file1.Delete()
        except:
            print("File not found")

        uploaded = self.drive_instance.CreateFile({model_title: self.model_name, "parents": [{"kind": "drive#fileLink", "id": parent_directory_id}]})
        uploaded.SetContentFile(self.model_path)
        uploaded.Upload()

        # Save training loss and accuracy
        parent_directory_id = '0B5fJkPjHLj3Jdkw5ZnFiY0lZV1U'
        file_list = self.drive_instance.ListFile(
            {'q': "\'" + parent_directory_id + "\'" + " in parents  and trashed=false"}).GetList()
        try:
            for file1 in file_list:
                if file1['title'] == hist_path:
                    file1.Delete()
        except:
            print("File not found")

        uploaded = self.drive_instance.CreateFile(
            {model_title: "Train history", "parents": [{"kind": "drive#fileLink", "id": parent_directory_id}]})
        uploaded.SetContentFile(self.history)
        uploaded.Upload()


