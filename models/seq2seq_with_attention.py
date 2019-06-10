from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input, Concatenate

from etc import settings
from .encoder_decoder import get_encoder_states_LSTM, get_decoder_outputs_LSTM, encoder_bi_LSTM, \
    decoder_for_bidirectional_encoder_LSTM
from .encoder_decoder import get_encoder_states_GRU_attention, get_decoder_outputs_GRU_attention, encoder_bi_GRU, \
    decoder_for_bidirectional_encoder_GRU
from .layers import AttentionLayer



def train_attention_seq2seq_model_GRU(mfcc_features, target_length, latent_dim):
    """
    :param mfcc_features:
    :param target_length:
    :param latent_dim:
    :return:
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_inputs")
    encoder_outputs, encoder_states = get_encoder_states_GRU_attention(mfcc_features=mfcc_features,
                                                                       encoder_inputs=encoder_inputs,
                                                                       latent_dim=latent_dim,
                                                                       return_sequences=True)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length), name="decoder_input")
    decoder_outputs, decoder_states = get_decoder_outputs_GRU_attention(target_length=target_length,
                                                        encoder_states=encoder_states,
                                                        decoder_inputs=decoder_inputs,
                                                        latent_dim=latent_dim)

    # Attention layer
    attn_layer = AttentionLayer(name='attention_layer')
    print(encoder_outputs)
    print(decoder_outputs)
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


def train_bidirectional_attention_seq2seq_model(mfcc_features, target_length, latent_dim):
    """
    :param mfcc_features:
    :param target_length:
    :param latent_dim:
    :return:
    """
    # Encoder training
    encoder_inputs = Input(shape=(settings.ENCODER_INPUT_MAX_LENGTH, mfcc_features), name="encoder_inputs")
    encoder_outputs, encoder_states = encoder_bi_LSTM(mfcc_features=mfcc_features,
                                                      encoder_inputs=encoder_inputs,
                                                      latent_dim=latent_dim,
                                                      return_sequences=True)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(settings.DECODER_INPUT_MAX_LENGTH, target_length), name="decoder_inputs")
    decoder_outputs = decoder_for_bidirectional_encoder_LSTM(target_length=target_length,
                                                             encoder_states=encoder_states,
                                                             decoder_inputs=decoder_inputs,
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
