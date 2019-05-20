import random
import numpy as np
from etc import settings
from utils import file_exists, get_files, load_pickle_data
from tensorflow.python.keras import models
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from .seq2seq_baseline import train_baseline_seq2seq_model, train_bidirectional_baseline_seq2seq_model
from .seq2seq_cnn_attention import train_cnn_seq2seq_model, train_cnn_attention_seq2seq_model, \
    train_cnn_bidirectional_attention_seq2seq_model
from .seq2seq_with_attention import train_attention_seq2seq_model, train_bidirectional_attention_seq2seq_model
from .model_callback import ModelSaver


class Seq2SeqModel():
    def __init__(self, latent_dim=300, epochs=50, model_architecture=5, data_generation=True, word_level=False):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.model_architecture = model_architecture
        self.data_generation = data_generation
        self.model_name = "architecture" + str(self.model_architecture) + ".h5"
        self.model_path = settings.TRAINED_MODELS_PATH + self.model_name
        self.mfcc_features_length = settings.MFCC_FEATURES_LENGTH
        if word_level:
            self.target_length = len(settings.WORD_SET)
        else:
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

    def train_model(self):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model_saver = ModelSaver(model_name=self.model_name, model_path=self.model_path,
                                 drive_instance=settings.DRIVE_INSTANCE)

        if self.data_generation:
            #generated_data = self._generate_timestep_dict(encoder_input_data, decoder_input_data, decoder_target_data)
            history = self.model.fit_generator(self.split_data_generator_dict(),
                                               steps_per_epoch=settings.TOTAL_SAMPLES_NUMBER,
                                               epochs=self.epochs,
                                               callbacks=[model_saver])

        else:
            pass
            #history = self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            #                         epochs=self.epochs,
            #                         validation_split=0.2,
            #                         callbacks=[model_saver])

    def _get_encoder_decoder_model_base_line(self):

        # Getting layers after training (updated weights)
        encoder_inputs = self.model.get_layer("encoder_inputs")
        decoder_inputs = self.model.get_layer("decoder_inputs")
        decoder_lstm2_layer = self.model.get_layer("decoder_lstm2_layer")
        decoder_lstm1_layer = self.model.get_layer("decoder_lstm1_layer")
        decoder_dense = self.model.get_layer("decoder_dense")

        # Creating encoder model
        encoder_model = Model(encoder_inputs, self.encoder_states)

        # Input shapes for 1st LSTM layer
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm1 = decoder_lstm1_layer(decoder_inputs, initial_state=decoder_states_inputs)

        # Outputs and states from final LSTM Layer
        decoder_outputs, state_h, state_c = decoder_lstm2_layer(decoder_lstm1)
        decoder_states = [state_h, state_c]

        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model

    def _data_generator(self, encoder_input, decoder_input, decoder_target):
        while True:
            index = random.randint(0, len(encoder_input) - 1)
            encoder_x = np.array([encoder_input[index]])
            decoder_x = np.array([decoder_input[index]])
            decoder_y = np.array([decoder_target[index]])

            yield [encoder_x, decoder_x], decoder_y


    def split_data_generator_dict_bis(self):
        audio_directory = settings.AUDIO_SPLIT_TRAIN_PATH
        audio_files = get_files(audio_directory)
        transcripts_directory = settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH
        transcript_files = get_files(transcripts_directory)
        visited_files = []
        data = self.get_data(audio_files[0], transcript_files[0])
        while True:
                #retrieving data
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


    def split_data_generator_dict(self):
        audio_directory = settings.AUDIO_SPLIT_TRAIN_PATH
        audio_files = get_files(audio_directory)
        transcripts_directory = settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH
        transcript_files = get_files(transcripts_directory)
        visited_files = []

        while True:
            for i, audio_file in enumerate(audio_files):
                #retrieving data

                data = self.get_data(audio_file, transcript_files[i])
                for pair_key in data:
                    #pair_key = random.choice(list(d.keys()))
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

    def get_data(self, audio_file, transcripts_file):
        encoder_input_data = load_pickle_data(audio_file)
        (decoder_input_data, decoder_target_data) = load_pickle_data(transcripts_file)
        data = self._generate_timestep_dict(encoder_input_data, decoder_input_data, decoder_target_data)
        return data

    def _generate_timestep_dict(self, encoder_input_data, decoder_input_data, decoder_target_data):
        generated_data = dict()
        for index, encoder_input in enumerate(encoder_input_data):
            key_pair = (len(encoder_input), len(decoder_input_data[index]))
            if not key_pair in generated_data:
                generated_data[key_pair] = []
            generated_data[key_pair].append([[encoder_input, decoder_input_data[index]], decoder_target_data[index]])

        return generated_data
