from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Concatenate
#from keras.layers import Dense, Input, Dropout, Concatenate
#from keras import Model
from .encoder_decoder import get_encoder_states, get_decoder_outputs, encoder_bilstm, \
    encoder_bilstm_GRU, decoder_for_bidirectional_encoder, decoder_for_bidirectional_encoder_GRU
from .encoder_decoder import get_encoder_states_GRU, get_decoder_outputs_GRU
from .layers import get_cnn_model
from .layers import AttentionLayer
from etc import settings

def train_cnn_seq2seq_model(mfcc_features, target_length, latent_dim, word_based):
    """
    trains Encoder/Decoder CNN based architecture and prepares encoder_model and decoder_model for prediction part
    :param audio_length: int
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """

    # getting CNN model
    cnn_input_shape = (None, mfcc_features)

    cnn_inputs = Input(shape=cnn_input_shape, name="encoder_inputs")
    cnn_model = get_cnn_model(cnn_input_shape)
    cnn_model_output_shape = cnn_model.layers[-1].output_shape[2]
    # Preparing Input shape for LSTM layer from CNN model
    cnn_output = cnn_model(cnn_inputs)
    encoder_states = get_encoder_states_GRU(mfcc_features=cnn_model_output_shape,
                                        encoder_inputs=cnn_output,
                                        latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length), name="decoder_input")
    decoder_outputs, states = get_decoder_outputs_GRU(target_length=target_length,
                                          encoder_states=encoder_states,
                                          decoder_inputs=decoder_inputs,
                                          latent_dim=latent_dim)

    if word_based:
        target_length = len(settings.CHARACTER_SET) + 1
        decoder_outputs = get_multi_output_dense(decoder_outputs, target_length=target_length)
    else:
        decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([cnn_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())

    return model, encoder_states


def train_cnn_attention_seq2seq_model( mfcc_features, target_length, latent_dim):
    """
    trains Encoder/Decoder CNN based architecture and prepares encoder_model and decoder_model for prediction part
    :param audio_length: int
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    #cnn_input_shape = (audio_length, mfcc_features)
    # getting CNN model
    cnn_inputs = Input(shape=(None, mfcc_features), name="encoder_inputs")
    cnn_model = get_cnn_model()

    # Preparing Input shape for LSTM layer from CNN model
    cnn_output = cnn_model(cnn_inputs)
    encoder_outputs, encoder_states = get_encoder_states(mfcc_features=mfcc_features,
                                        encoder_inputs=cnn_output,
                                        latent_dim=latent_dim,
                                        return_sequences=True)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length), name="decoder_inputs")
    decoder_outputs = get_decoder_outputs(target_length=target_length,
                                          encoder_states=encoder_states,
                                          decoder_inputs=decoder_inputs,
                                          latent_dim=latent_dim)

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    # Dense Output Layers
    decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Dense Output Layers
    #decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
    #decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([cnn_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())

    return model, encoder_states


def train_cnn_bidirectional_attention_seq2seq_model(mfcc_features, target_length, latent_dim, word_level):
    """
    trains Encoder/Decoder CNN based architecture and prepares encoder_model and decoder_model for prediction part
    :param audio_length: int
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    cnn_input_shape = (None, mfcc_features)

    cnn_inputs = Input(shape=cnn_input_shape, name="encoder_inputs")
    cnn_model = get_cnn_model(cnn_input_shape)
    cnn_model_output_shape = cnn_model.layers[-1].output_shape[2]
    # Preparing Input shape for LSTM layer from CNN model
    cnn_output = cnn_model(cnn_inputs)
    encoder_states = encoder_bilstm_GRU(mfcc_features=cnn_model_output_shape,
                                        encoder_inputs=cnn_output,
                                        latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length), name="decoder_inputs")
    decoder_outputs, decoder_states = decoder_for_bidirectional_encoder_GRU(target_length=target_length,
                                                        encoder_states=encoder_states,
                                                        decoder_inputs=decoder_inputs,
                                                        latent_dim=latent_dim)

    decoder_dropout = Dropout(0.2)
    decoder_outputs = decoder_dropout(decoder_outputs)
    if word_level:
        target_length = len(settings.CHARACTER_SET) + 1
        decoder_outputs = get_multi_output_dense(decoder_outputs, target_length)
    else:
        decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([cnn_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())

    return model, encoder_states


def get_multi_output_dense(decoder_outputs, target_length):
    dense_layers = []

    for i in range(0, settings.LONGEST_WORD_LENGTH):
        decoder_dense = Dense(target_length, activation='softmax', name="dense"+str(i))
        new_decoder_output = decoder_dense(decoder_outputs)
        dense_layers.append(new_decoder_output)
    return dense_layers
