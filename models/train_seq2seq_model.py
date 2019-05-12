import matplotlib.pyplot as plt
from etc import settings
from utils import file_exists
from tensorflow.python.keras import models
from .seq2seq_baseline import train_baseline_seq2seq_model, train_bidirectional_baseline_seq2seq_model
from .seq2seq_cnn_attention import train_cnn_seq2seq_model, train_cnn_attention_seq2seq_model, train_cnn_bidirectional_attention_seq2seq_model
from .seq2seq_with_attention import train_attention_seq2seq_model, train_bidirectional_attention_seq2seq_model
from .model_callback import ModelSaver


def train_model(encoder_input_data, decoder_input_data,decoder_target_data,
                latent_dim=256, batch_size=64, epochs=70, model_architecture=1):
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
        length = encoder_input_data.shape[1]
        model, encoder_states = train_cnn_seq2seq_model(audio_length=length,
                                                                  mfcc_features=mfcc_features_length,
                                                                  target_length=target_length,
                                                                  batch_size=batch_size,
                                                                  latent_dim=latent_dim)
    elif model_architecture == 6:
        length = encoder_input_data.shape[1]
        model, encoder_states = train_cnn_attention_seq2seq_model(audio_length=length,
                                                                  mfcc_features=mfcc_features_length,
                                                                  target_length=target_length,
                                                                  batch_size=batch_size,
                                                                  latent_dim=latent_dim)

    else:
        length = encoder_input_data.shape[1]
        model, encoder_states = train_cnn_bidirectional_attention_seq2seq_model(audio_length=length,
                                                                  mfcc_features=mfcc_features_length,
                                                                  target_length=target_length,
                                                                  batch_size=batch_size,
                                                                  latent_dim=latent_dim)

    # Training model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model_name = "architecture" + str(model_architecture) + ".h5"
    model_path = settings.TRAINED_MODELS_PATH+model_name

    model_saver = ModelSaver(model_name=model_name, model_path=model_path, drive_instance=settings.DRIVE_INSTANCE)

    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[model_saver])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return model, encoder_states











