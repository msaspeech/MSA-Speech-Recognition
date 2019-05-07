from keras.callbacks import Callback


class ModelSaver(Callback):

    def __init__(self, model_name, model_path, drive_instance):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.drive_instance = drive_instance

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path)
        model_title = self.model_name.split(".h5")[0]
        parent_directory_id = '0B5fJkPjHLj3Jdkw5ZnFiY0lZV1U'
        uploaded = self.drive_instance.CreateFile({model_title: self.model_name, "parents": [{"kind": "drive#fileLink", "id": parent_directory_id}]})
        uploaded.SetContentFile(self.model_path)
        uploaded.Upload()

