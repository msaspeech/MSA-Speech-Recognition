from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import TimeDistributed, Dense, Input, CuDNNLSTM, Concatenate, LSTM
from .layers import AttentionLayer, get_cnn_model

from etc import settings


def get_encoder_states(mfcc_features, encoder_inputs, latent_dim, return_sequences=False):
    encoder = CuDNNLSTM(latent_dim,
                        batch_input_shape=(1, None, mfcc_features),
                        stateful=False,
                        return_state=True,
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
                                    batch_input_shape=(1, None, target_length),
                                    stateful=False,
                                    return_sequences=True,
                                    return_state=False,
                                    name="decoder_lstm1_layer")

    decoder_lstm1 = decoder_lstm1_layer(decoder_inputs, initial_state=encoder_states)

    # Second LSTM Layer
    decoder_lstm2_layer = CuDNNLSTM(latent_dim,
                                    stateful=False,
                                    return_sequences=True,
                                    return_state=True,
                                    name="decoder_lstm_2layer")
    decoder_outputs, _, _ = decoder_lstm2_layer(decoder_lstm1)
    return decoder_outputs


def train_baseline_seq2seq_model(mfcc_features=40, target_length=42, latent_dim=512):
    """
    trains Encoder/Decoder architecture and prepares encoder_model and decoder_model for prediction part
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_input")
    print(encoder_inputs)
    encoder_states = get_encoder_states(mfcc_features=mfcc_features,
                                        encoder_inputs=encoder_inputs,
                                        latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length), name="decoder_inputs")
    decoder_outputs = get_decoder_outputs(target_length=target_length,
                                          encoder_states=encoder_states,
                                          decoder_inputs=decoder_inputs,
                                          latent_dim=latent_dim)

    # Dense Output Layers
    decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model, encoder_states


def train_attention_seq2seq_model(mfcc_features=40, target_length=42, latent_dim=512):
    """
    :param mfcc_features:
    :param target_length:
    :param latent_dim:
    :return:
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_inputs")
    encoder_outputs, encoder_states = get_encoder_states(mfcc_features=mfcc_features,
                                                         encoder_inputs=encoder_inputs,
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
    dense = Dense(target_length, activation='softmax', name="decoder_dense")
    # dense_time = TimeDistributed(dense, name='time_distributed_layer')
    decoder_pred = dense(decoder_concat_input)

    # Generating Keras Model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    return model, encoder_states


def train_cnn_attention_seq2seq_model(audio_length, mfcc_features=40, target_length=42, latent_dim=512):
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
                                        latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length), name="decoder_inputs")
    decoder_outputs = get_decoder_outputs(target_length=target_length,
                                          encoder_states=encoder_states,
                                          decoder_inputs=decoder_inputs,
                                          latent_dim=latent_dim)

    # Dense Output Layers
    decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([cnn_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())

    return model, encoder_states


def train_model(encoder_input_data, decoder_input_data,decoder_target_data,
                latent_dim=512, model_architecture=1, batch_size=64, epochs=70):
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
                                                             latent_dim=latent_dim)

    elif model_architecture == 2:
        model, encoder_states = train_attention_seq2seq_model(mfcc_features=mfcc_features_length,
                                                              target_length=target_length,
                                                              latent_dim=latent_dim)

    else:
        length = encoder_input_data.shape[1]
        model, encoder_states = train_cnn_attention_seq2seq_model(audio_length=length,
                                                                  mfcc_features=mfcc_features_length,
                                                                  target_length=target_length,
                                                                  latent_dim=latent_dim)

    # Training model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)

    model_name = "architecture"+str(model_architecture)
    model.save(model_name)

    return model, encoder_states











