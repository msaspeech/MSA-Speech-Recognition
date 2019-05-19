import random
import numpy as np
from etc import settings
from utils import file_exists
from tensorflow.python.keras import models
from .seq2seq_baseline import train_baseline_seq2seq_model, train_bidirectional_baseline_seq2seq_model
from .seq2seq_cnn_attention import train_cnn_seq2seq_model, train_cnn_attention_seq2seq_model, \
    train_cnn_bidirectional_attention_seq2seq_model
from .seq2seq_with_attention import train_attention_seq2seq_model, train_bidirectional_attention_seq2seq_model
from .model_callback import ModelSaver


class Seq2SeqModel():
    def __init__(self, latent_dim=300, epochs=50, model_architecture=5, data_generation=True):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.model_architecture = model_architecture
        self.data_generation = data_generation
        self.model_name = "architecture" + str(self.model_architecture) + ".h5"
        self.model_path = settings.TRAINED_MODELS_PATH + self.model_name
        self.mfcc_features_length = settings.MFCC_FEATURES_LENGTH
        self.target_length = len(settings.CHARACTER_SET)
        self.model = None
        self.encoder_states = None
        self._load_model()

    def _load_model(self):
        if file_exists(self.model_path):
            self.model = models.load_model(self.model_path)
        else:
            if self.model_architecture == 1:
                self.model, self.encoder_states = train_baseline_seq2seq_model(mfcc_features=self.mfcc_features_length,
                                                                               target_length=self.target_length,
                                                                               latent_dim=self.latent_dim)
            elif self.model_architecture == 2:
                self.model, self.encoder_states = train_bidirectional_baseline_seq2seq_model(mfcc_features=self.mfcc_features_length,
                                                                                             target_length=self.target_length,
                                                                                             latent_dim=self.latent_dim)

            elif self.model_architecture == 3:
                self.model, self.encoder_states = train_attention_seq2seq_model(mfcc_features=self.mfcc_features_length,
                                                                                target_length=self.target_length,
                                                                                latent_dim=self.latent_dim)
            elif self.model_architecture == 4:
                self.model, self.encoder_states = train_bidirectional_attention_seq2seq_model(
                    mfcc_features=self.mfcc_features_length,
                    target_length=self.target_length,
                    latent_dim=self.latent_dim)

            elif self.model_architecture == 5:
                self.model, self.encoder_states = train_cnn_seq2seq_model(mfcc_features=self.mfcc_features_length,
                                                                          target_length=self.target_length,
                                                                          latent_dim=self.latent_dim)
            elif self.model_architecture == 6:
                self.model, self.encoder_states = train_cnn_attention_seq2seq_model(mfcc_features=self.mfcc_features_length,
                                                                                    target_length=self.target_length,
                                                                                    latent_dim=self.latent_dim)

            else:
                self.model, self.encoder_states = train_cnn_bidirectional_attention_seq2seq_model(mfcc_features=self.mfcc_features_length,
                                                                                                  target_length=self.target_length,
                                                                                                  latent_dim=self.latent_dim)

    def train_model(self, encoder_input_data, decoder_input_data, decoder_target_data):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model_saver = ModelSaver(model_name=self.model_name, model_path=self.model_path,
                                 drive_instance=settings.DRIVE_INSTANCE)

        if self.data_generation:
            generated_data = self._generate_timestep_dict(encoder_input_data, decoder_input_data, decoder_target_data)
            history = self.model.fit_generator(self._data_generator_dict(generated_data),
                                               steps_per_epoch=len(encoder_input_data),
                                               epochs=self.epochs,
                                               callbacks=[model_saver])

        else:
            history = self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                     epochs=self.epochs,
                                     validation_split=0.2,
                                     callbacks=[model_saver])

    def _data_generator(self, encoder_input, decoder_input, decoder_target):
        while True:
            index = random.randint(0, len(encoder_input) - 1)
            encoder_x = np.array([encoder_input[index]])
            decoder_x = np.array([decoder_input[index]])
            decoder_y = np.array([decoder_target[index]])

            yield [encoder_x, decoder_x], decoder_y

    def _data_generator_dict(self, data):

        while True:
            pair_key = random.choice(list(data.keys()))
            output = data[pair_key]
            encoder_x = []
            decoder_x = []
            decoder_y = []
            for element in output:
                encoder_x.append(element[0][0])
                decoder_x.append(element[0][1])
                decoder_y.append(element[1])

            encoder_x = np.array(encoder_x)
            decoder_x = np.array(decoder_x)
            decoder_y = np.array(decoder_y)

            yield [encoder_x, decoder_x], decoder_y

    def _generate_timestep_dict(self, encoder_input_data, decoder_input_data, decoder_target_data):
        generated_data = dict()
        for index, encoder_input in enumerate(encoder_input_data):
            key_pair = (len(encoder_input), len(decoder_input_data[index]))
            if not key_pair in generated_data:
                generated_data[key_pair] = []
            generated_data[key_pair].append([[encoder_input, decoder_input_data[index]], decoder_target_data[index]])

        return generated_data
