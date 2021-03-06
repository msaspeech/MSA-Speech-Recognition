from keras import Model
from keras.layers import Dense, Input, Concatenate

from etc import ENCODER_INPUT_MAX_LENGTH, DECODER_INPUT_MAX_LENGTH
from .encoder_decoder import get_encoder_states, get_decoder_outputs, encoder_bilstm, decoder_for_bidirectional_encoder
from .layers import AttentionLayer


def train_attention_seq2seq_model(mfcc_features, target_length, latent_dim, batch_size):
    """
    :param mfcc_features:
    :param target_length:
    :param latent_dim:
    :return:
    """
    # Encoder training
    encoder_inputs = Input(shape=(ENCODER_INPUT_MAX_LENGTH, mfcc_features), name="encoder_inputs")
    encoder_outputs, encoder_states = get_encoder_states(mfcc_features=mfcc_features,
                                                         encoder_inputs=encoder_inputs,
                                                         latent_dim=latent_dim,
                                                         batch_size=batch_size,
                                                         return_sequences=True)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(DECODER_INPUT_MAX_LENGTH, target_length), name="decoder_inputs")
    decoder_outputs = get_decoder_outputs(target_length=target_length,
                                          encoder_states=encoder_states,
                                          decoder_inputs=decoder_inputs,
                                          batch_size=batch_size,
                                          latent_dim=latent_dim)

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    # Dense Output Layers
    dense = Dense(target_length, activation='softmax', name="decoder_dense")
    # dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense(decoder_concat_input)

    # Generating Keras Model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    print(model.summary())
    return model, encoder_states


def train_bidirectional_attention_seq2seq_model(mfcc_features, target_length, latent_dim, batch_size):
    """
    :param mfcc_features:
    :param target_length:
    :param latent_dim:
    :return:
    """
    # Encoder training
    encoder_inputs = Input(shape=(ENCODER_INPUT_MAX_LENGTH, mfcc_features), name="encoder_inputs")
    encoder_outputs, encoder_states = encoder_bilstm(mfcc_features=mfcc_features,
                                                         encoder_inputs=encoder_inputs,
                                                         latent_dim=latent_dim,
                                                         batch_size=batch_size,
                                                         return_sequences=True)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(DECODER_INPUT_MAX_LENGTH, target_length), name="decoder_inputs")
    decoder_outputs = decoder_for_bidirectional_encoder(target_length=target_length,
                                          encoder_states=encoder_states,
                                          decoder_inputs=decoder_inputs,
                                          batch_size=batch_size,
                                          latent_dim=latent_dim)

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

    # Dense Output Layers
    dense = Dense(target_length, activation='softmax', name="decoder_dense")
    # dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense(decoder_concat_input)

    # Generating Keras Model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    print(model.summary())
    return model, encoder_states
