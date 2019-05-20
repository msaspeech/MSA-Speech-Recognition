from tensorflow.python.keras.layers import CuDNNLSTM, Bidirectional, Concatenate, LSTM


def get_encoder_states(mfcc_features, encoder_inputs, latent_dim, return_sequences=False):
    encoder = CuDNNLSTM(latent_dim,
                        input_shape=(None, mfcc_features),
                        stateful=False,
                        return_sequences=return_sequences,
                        return_state=True,
                        kernel_constraint=None,
                        kernel_regularizer=None,
                        recurrent_initializer='glorot_uniform')
    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    if return_sequences:
        return encoder_outputs, encoder_states
    else:
        return encoder_states


def get_decoder_outputs(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_lstm1_layer = CuDNNLSTM(latent_dim,
                                    input_shape=(None, target_length),
                                    return_sequences=True,
                                    return_state=False,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm1_layer")
    decoder_lstm1 = decoder_lstm1_layer(decoder_inputs, initial_state=encoder_states)

    # Second LSTM Layer
    decoder_lstm2_layer = CuDNNLSTM(latent_dim,
                                    stateful=False,
                                    return_sequences=True,
                                    return_state=True,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm2_layer")

    decoder_outputs, _, _ = decoder_lstm2_layer(decoder_lstm1)
    return decoder_outputs


def encoder_bilstm(mfcc_features, encoder_inputs, latent_dim, return_sequences=False):
    encoder = Bidirectional(CuDNNLSTM(latent_dim,
                            input_shape=(None, mfcc_features),
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


def decoder_for_bidirectional_encoder(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_lstm1_layer = CuDNNLSTM(latent_dim*2,
                                    input_shape=(None, target_length),
                                    return_sequences=True,
                                    return_state=False,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm1_layer")
    decoder_lstm1 = decoder_lstm1_layer(decoder_inputs, initial_state=encoder_states)

    # Second LSTM Layer
    decoder_lstm2_layer = CuDNNLSTM(latent_dim*2,
                                    stateful=False,
                                    return_sequences=True,
                                    return_state=True,
                                    kernel_constraint=None,
                                    kernel_regularizer=None,
                                    name="decoder_lstm_2layer")
    decoder_outputs, _, _ = decoder_lstm2_layer(decoder_lstm1)
    return decoder_outputs
