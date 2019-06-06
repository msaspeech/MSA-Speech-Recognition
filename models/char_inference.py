from etc import settings


class Char_Inference():

    def __init__(self):
        self.model = models.load_model(model_path)
        self.encoder_states = None
        self.latent_dim = latent_dim
        general_info = load_pickle_data(settings.DATASET_INFERENCE_INFORMATION_PATH)
        settings.MFCC_FEATURES_LENGTH = general_info[0]
        settings.CHARACTER_SET = general_info[2]
        self.character_set = settings.CHARACTER_SET
        self.word_level = word_level
        self.encoder_model, self.decoder_model = self._get_encoder_decoder_model_baseline()