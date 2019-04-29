from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input

from .encoder_decoder import get_encoder_states, get_decoder_outputs
from .layers import get_cnn_model


def train_cnn_attention_seq2seq_model(audio_length, mfcc_features, target_length, batch_size, latent_dim):
    """
    trains Encoder/Decoder CNN based architecture and prepares encoder_model and decoder_model for prediction part
    :param audio_length: int
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    cnn_input_shape = (audio_length, mfcc_features)
    # getting CNN model
    cnn_inputs = Input(shape=cnn_input_shape, name="encoder_inputs")
    cnn_model = get_cnn_model(cnn_input_shape)

    # Preparing Input shape for LSTM layer from CNN model
    cnn_output = cnn_model(cnn_inputs)
    encoder_states = get_encoder_states(mfcc_features=mfcc_features,
                                        encoder_inputs=cnn_output,
                                        batch_size=batch_size,
                                        latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length), name="decoder_inputs")
    decoder_outputs = get_decoder_outputs(target_length=target_length,
                                          encoder_states=encoder_states,
                                          decoder_inputs=decoder_inputs,
                                          batch_size=batch_size,
                                          latent_dim=latent_dim)

    # Dense Output Layers
    decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([cnn_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())

    return model, encoder_states
