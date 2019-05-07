from keras.callbacks import Callback


class ModelSaver(Callback):

    def __init__(self, model_name, model_path, drive_instance):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.drive_instance = drive_instance
        if self.model is not None :
            self.model.save(model_path)
        else:
            print("model is None")

    def on_epoch_end(self, epoch, logs=None):
        model_title = self.model_name.split(".h5")[0]
        uploaded = self.drive_instance.CreateFile({model_title: self.model_name})
        uploaded.SetContentFile(self.model_path)
        uploaded.Upload()
