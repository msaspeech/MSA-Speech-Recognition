from livelossplot.keras import PlotLossesCallback

from etc import settings
from .seq2seq_baseline import train_baseline_seq2seq_model
from .seq2seq_cnn_attention import train_cnn_attention_seq2seq_model
from .seq2seq_with_attention import train_attention_seq2seq_model


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
        model, encoder_states = train_attention_seq2seq_model(mfcc_features=mfcc_features_length,
                                                              target_length=target_length,
                                                              batch_size=batch_size,
                                                              latent_dim=latent_dim)

    else:
        length = encoder_input_data.shape[1]
        model, encoder_states = train_cnn_attention_seq2seq_model(audio_length=length,
                                                                  mfcc_features=mfcc_features_length,
                                                                  target_length=target_length,
                                                                  batch_size=batch_size,
                                                                  latent_dim=latent_dim)

    # Training model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[PlotLossesCallback()])

    model_name = "trained_models/architecture"+str(model_architecture)+".h5"
    model.save(model_name)

    return model, encoder_states











