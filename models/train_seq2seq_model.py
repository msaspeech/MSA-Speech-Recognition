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


def train_model(encoder_input_data, decoder_input_data, decoder_target_data,
                latent_dim=256, batch_size=64, epochs=70, model_architecture=1, data_generation=False):
    """
    Choosing the architecture and running a training
    :param encoder_input_data: Numpy 3dArray
    :param decoder_input_data: Numpy 3dArray
    :param decoder_target_data: Numpy 3dArray
    :param latent_dim: int
    :param model_architecture: int
    :param batch_size: int
    :param epochs: int
    """
    mfcc_features_length = settings.MFCC_FEATURES_LENGTH
    target_length = len(settings.CHARACTER_SET)

    model_name = "architecture" + str(model_architecture) + ".h5"
    model_path = settings.TRAINED_MODELS_PATH + model_name

    if file_exists(model_path):
        model = models.load_model(model_path)
    else:
        if model_architecture == 1:
            model, encoder_states = train_baseline_seq2seq_model(mfcc_features=mfcc_features_length,
                                                                 target_length=target_length,
                                                                 batch_size=batch_size,
                                                                 latent_dim=latent_dim)
        elif model_architecture == 2:
            model, encoder_states = train_bidirectional_baseline_seq2seq_model(mfcc_features=mfcc_features_length,
                                                                               target_length=target_length,
                                                                               batch_size=batch_size,
                                                                               latent_dim=latent_dim)

        elif model_architecture == 3:
            model, encoder_states = train_attention_seq2seq_model(mfcc_features=mfcc_features_length,
                                                                  target_length=target_length,
                                                                  batch_size=batch_size,
                                                                  latent_dim=latent_dim)
        elif model_architecture == 4:
            model, encoder_states = train_bidirectional_attention_seq2seq_model(mfcc_features=mfcc_features_length,
                                                                                target_length=target_length,
                                                                                batch_size=batch_size,
                                                                                latent_dim=latent_dim)

        elif model_architecture == 5:
            #length = encoder_input_data.shape[1]
            model, encoder_states = train_cnn_seq2seq_model(
                                                            mfcc_features=mfcc_features_length,
                                                            target_length=target_length,
                                                            batch_size=batch_size,
                                                            latent_dim=latent_dim)
        elif model_architecture == 6:
            #length = encoder_input_data.shape[1]
            model, encoder_states = train_cnn_attention_seq2seq_model(
                                                                      mfcc_features=mfcc_features_length,
                                                                      target_length=target_length,
                                                                      batch_size=batch_size,
                                                                      latent_dim=latent_dim)

        else:
            #length = encoder_input_data.shape[1]
            model, encoder_states = train_cnn_bidirectional_attention_seq2seq_model(
                                                                                    mfcc_features=mfcc_features_length,
                                                                                    target_length=target_length,
                                                                                    batch_size=batch_size,
                                                                                    latent_dim=latent_dim)

        # Training model
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model_saver = ModelSaver(model_name=model_name, model_path=model_path, drive_instance=settings.DRIVE_INSTANCE)

    if data_generation:
        generated_data = generate_timestep_dict(encoder_input_data, decoder_input_data, decoder_target_data)
        history = model.fit_generator(data_generator_dict(generated_data),
                                      steps_per_epoch=len(encoder_input_data),
                                      epochs=epochs,
                                      callbacks=[model_saver])

    else:
        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            callbacks=[model_saver])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    # summarize history for loss
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.show()

    return model, encoder_states


def data_generator(encoder_input, decoder_input, decoder_target):
    while True:

        index = random.randint(0, len(encoder_input) - 1)
        encoder_x = np.array([encoder_input[index]])
        decoder_x = np.array([decoder_input[index]])
        decoder_y = np.array([decoder_target[index]])

        yield [encoder_x, decoder_x], decoder_y


def data_generator_dict(data):

    while True :
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


def generate_timestep_dict(encoder_input_data, decoder_input_data, decoder_target_data):
    generated_data = dict()
    for index, encoder_input in enumerate(encoder_input_data):
        key_pair = (len(encoder_input), len(decoder_input_data[index]))
        if not key_pair in generated_data:
            generated_data[key_pair] = []
        generated_data[key_pair].append([[encoder_input, decoder_input_data[index]], decoder_target_data[index]])

    return generated_data

