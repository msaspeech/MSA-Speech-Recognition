from tensorflow.python.keras.callbacks import Callback, History
import matplotlib.pyplot as plt
from etc import MODEL_HISTORY_PLOTS, ENCODER_STATES_PATH
from utils import generate_pickle_file


class ModelSaver(Callback):

    def __init__(self, model_name, model_path, encoder_states, drive_instance):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.encoder_states = encoder_states
        self.drive_instance = drive_instance
        self.history = History()

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path)
        model_title = self.model_name.split(".h5")[0]

        # Saving encoder states
        path = ENCODER_STATES_PATH + model_title + ".pkl"
        print("ENCODER STATES")
        print(self.encoder_states)
        encoder_states = [self.encoder_states]
        generate_pickle_file(encoder_states, path)

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



        #Model saving

        #plt.plot(self.history.history['acc'])
        # plt.plot(history.history['val_acc'])
        #plt.title('model accuracy')
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train'], loc='upper left')
        #save_path =  MODEL_HISTORY_PLOTS+self.model_name.split(".h5")[0]+"_train_accuracy.png"
        #plt.savefig(save_path)

