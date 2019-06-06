from etc import settings
from utils import load_pickle_data
from tensorflow.python.keras import models
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input

class Word_Inference():

    def __init__(self, model_path, latent_dim):
        self.model = models.load_model(model_path)
        self.encoder_states = None
        self.latent_dim = latent_dim

        # Getting dataset and training information
        general_info = load_pickle_data(settings.DATASET_WORD_INFERENCE_INFORMATION_PATH)
        settings.MFCC_FEATURES_LENGTH = general_info[0]
        settings.TOTAL_SAMPLES_NUMBER = general_info[1]
        settings.WORD_SET = general_info[2]
        settings.LONGEST_WORD_LENGTH = general_info[3]
        settings.CHARACTER_SET = general_info[4]
        settings.WORD_TARGET_LENGTH = general_info[5]

        self.encoder_model = None
        self.decoder_model = None

    def get_encoder_decoder_baseline(self):

        # Getting encoder model
        encoder_inputs = self.model.get_layer("encoder_input").input
        [h] = self.model.get_layer("encoder_gru_layer").output[0]
        self.encoder_states = [h]

        encoder_model = Model(encoder_inputs, self.encoder_states)

        # Getting decoder model

        decoder_inputs = self.model.get_layer("decoder_input").input

        decoder_gru1_layer = self.model.get_layer("decoder_gru1_layer")
        decoder_gru2_layer = self.model.get_layer("decoder_gru2_layer")
        decoder_gru3_layer = self.model.get_layer("decoder_gru2_layer")
        decoder_dense_layers = []
        for i in range(0, settings.LONGEST_WORD_LENGTH):
            layer_name = "decoder_dense"+str(i)
            decoder_dense_layers.append(self.model.get_layer(layer_name))

        decoder_state_input_h = Input(shape=(self.latent_dim,))


    def get_encoder_decoder_cnn(self):
        pass

    def get_encoder_decoder_bi_baseline(self):
        pass

    def get_encoder_decoder_bi_cnn(self):
        pass