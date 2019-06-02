from scipy.optimize._trustregion_constr.canonical_constraint import initial_constraints_as_canonical
from tensorflow.python.keras.layers import CuDNNLSTM, Bidirectional, Concatenate, LSTM, GRU, CuDNNGRU
#from keras.layers import CuDNNLSTM, Bidirectional, Concatenate, LSTM


def get_encoder_states(mfcc_features, encoder_inputs, latent_dim, return_sequences=False):
    encoder = LSTM(latent_dim,
                        stateful=False,
                        return_sequences=return_sequences,
                        return_state=True,
                        kernel_constraint=None,
                        kernel_regularizer=None,
                        recurrent_initializer='glorot_uniform',
                        name="encoder_lstm_layer")
    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    if return_sequences:
        return encoder_outputs, encoder_states
    else:
        return encoder_states


def get_encoder_states_GRU(mfcc_features, encoder_inputs, latent_dim, return_sequences=False):
    encoder = CuDNNGRU(latent_dim,
                        stateful=False,
                        return_sequences=return_sequences,
                        return_state=True,
                        kernel_constraint=None,
                        kernel_regularizer=None,
                        name="encoder_lstm_layer")
    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, state_h= encoder(encoder_inputs)

    encoder_states = [state_h]
    if return_sequences:
        return encoder_outputs, encoder_states
    else:
        return encoder_states


def get_decoder_outputs(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_lstm1_layer = LSTM(latent_dim,
                                    return_sequences=True,
                                    return_state=True,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm1_layer")
    decoder_outputs, state_h, state_c = decoder_lstm1_layer(decoder_inputs, initial_state=encoder_states)

    decoder_states = [state_h, state_c]

    return decoder_outputs, decoder_states


def get_decoder_outputs_GRU(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_lstm1_layer = CuDNNGRU(latent_dim,
                                    return_sequences=True,
                                    return_state=True,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm1_layer")
    decoder_outputs, state_h= decoder_lstm1_layer(decoder_inputs, initial_state=encoder_states)

    decoder_lstm2_layer = CuDNNGRU(latent_dim,
                                    return_sequences=True,
                                    return_state=True,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm2_layer")
    decoder_outputs, state_h = decoder_lstm2_layer(decoder_outputs, initial_state=state_h)

    decoder_lstm3_layer = CuDNNGRU(latent_dim,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_constraint=None,
                                   kernel_regularizer=None,
                                   name="decoder_lstm3_layer")
    decoder_outputs, state_h = decoder_lstm3_layer(decoder_outputs, initial_state=state_h)

    decoder_states = [state_h]

    return decoder_outputs, decoder_states


def encoder_bilstm(mfcc_features, encoder_inputs, latent_dim, return_sequences=False):
    encoder = Bidirectional(LSTM(latent_dim,
                            stateful=False,
                            return_sequences=return_sequences,
                            return_state=True,
                            kernel_constraint=None,
                            kernel_regularizer=None,
                            recurrent_initializer='glorot_uniform'))
    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    if return_sequences:
        return encoder_outputs, encoder_states
    else:
        return encoder_states


def encoder_bilstm_GRU(mfcc_features, encoder_inputs, latent_dim, return_sequences=False):
    encoder = Bidirectional( CuDNNGRU(latent_dim,
                            input_shape=(None, mfcc_features),
                            stateful=False,
                            return_sequences=return_sequences,
                            return_state=True,
                            kernel_constraint=None,
                            kernel_regularizer=None,
                            recurrent_initializer='glorot_uniform'))
    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, forward_h, backward_h = encoder(encoder_inputs)
    state_h = Concatenate()([forward_h, backward_h])
    #state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h]
    if return_sequences:
        return encoder_outputs, encoder_states
    else:
        return encoder_states


def decoder_for_bidirectional_encoder(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_lstm1_layer = LSTM(latent_dim*2,
                                    input_shape=(None, target_length),
                                    return_sequences=True,
                                    return_state=False,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm1_layer")
    decoder_lstm1 = decoder_lstm1_layer(decoder_inputs, initial_state=encoder_states)

    # Second LSTM Layer
    decoder_lstm2_layer = LSTM(latent_dim*2,
                                    stateful=False,
                                    return_sequences=True,
                                    return_state=True,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm_2layer")
    decoder_outputs, _, _ = decoder_lstm2_layer(decoder_lstm1)
    return decoder_outputs


def decoder_for_bidirectional_encoder_GRU(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_lstm1_layer = CuDNNGRU(latent_dim*2,
                                    input_shape=(None, target_length),
                                    return_sequences=True,
                                    return_state=True,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm1_layer")
    decoder_lstm1, state_h = decoder_lstm1_layer(decoder_inputs, initial_state=encoder_states)

    # Second LSTM Layer
    decoder_lstm2_layer = CuDNNGRU(latent_dim*2,
                                    stateful=False,
                                    return_sequences=True,
                                    return_state=True,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm_2layer")
    decoder_outputs, state_h = decoder_lstm2_layer(decoder_lstm1, initial_state=state_h)

    decoder_lstm3_layer = CuDNNGRU(latent_dim * 2,
                                   stateful=False,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_constraint=None,
                                   kernel_regularizer=None,
                                   name="decoder_lstm_3layer")
    decoder_outputs, state_h = decoder_lstm3_layer(decoder_outputs, initial_state=state_h)

    decoder_states = [state_h]

    return decoder_outputs, decoder_states
