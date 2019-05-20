#from tensorflow.python.keras import Model
#from tensorflow.python.keras.layers import Dense, Input
from keras.layers import Dense, Input
from keras import Model
from .encoder_decoder import get_encoder_states, get_decoder_outputs, encoder_bilstm, decoder_for_bidirectional_encoder


def train_baseline_seq2seq_model(mfcc_features, target_length, latent_dim):
    """
    trains Encoder/Decoder architecture and prepares encoder_model and decoder_model for prediction part
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_input")
    encoder_states = get_encoder_states(mfcc_features=mfcc_features,
                                        encoder_inputs=encoder_inputs,
                                        latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length), name="decoder_input")
    #masked_inputs = Masking(mask_value=0,)(decoder_inputs)
    decoder_outputs = get_decoder_outputs(target_length=target_length,
                                          encoder_states=encoder_states,
                                          decoder_inputs=decoder_inputs,
                                          latent_dim=latent_dim)

    # Dense Output Layers
    decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
   # print(model.summary())
    return model, encoder_states


def train_bidirectional_baseline_seq2seq_model(mfcc_features, target_length, latent_dim):
    """
    trains Encoder/Decoder architecture and prepares encoder_model and decoder_model for prediction part
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_input")
    encoder_states = encoder_bilstm(mfcc_features=mfcc_features,
                                        encoder_inputs=encoder_inputs,
                                        latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.

    decoder_inputs = Input(shape=(None, target_length), name="decoder_input")
    decoder_outputs = decoder_for_bidirectional_encoder(target_length=target_length,
                                          encoder_states=encoder_states,
                                          decoder_inputs=decoder_inputs,
                                          latent_dim=latent_dim)

    # Dense Output Layers
    decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())
    return model, encoder_states

