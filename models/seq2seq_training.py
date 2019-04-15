from .conv_layers import get_cnn_model
from tensorflow.python.keras.layers import RepeatVector, TimeDistributed, Dense, Input, CuDNNLSTM
from tensorflow.python.keras import Model


def get_encoder_states(mfcc_features, encoder_inputs, latent_dim):
    encoder = CuDNNLSTM(latent_dim,
                        batch_input_shape=(None, None, mfcc_features),
                        stateful=False,
                        return_state=True,
                        recurrent_initializer='glorot_uniform')

    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    return encoder_states


def get_decoder_outputs(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_lstm_1_layer = CuDNNLSTM(latent_dim,
                                     batch_input_shape=(None, None, target_length),
                                     stateful=False,
                                     return_state=False,
                                     dropout=0.2,
                                     recurrent_dropout=0.2)
    decoder_lstm1 = decoder_lstm_1_layer(decoder_inputs, initial_state=encoder_states)
    # Second LSTM Layer
    decoder_lstm_2_layer = CuDNNLSTM(latent_dim,
                                     stateful=False,
                                     return_sequences=True,
                                     return_state=True,
                                     dropout=0.2,
                                     recurrent_dropout=0.2)
    decoder_outputs, _, _ = decoder_lstm_2_layer(decoder_lstm1)
    return decoder_outputs, (decoder_lstm_1_layer, decoder_lstm_2_layer)


def train_baseline_seq2seq_model(encoder_input_data, decoder_input_data, decoder_target_data,
                                 mfcc_features=40, target_length=40, latent_dim=512,
                                 batch_size=64, epochs=70):
    """

    :param encoder_input_data:
    :param decoder_input_data:
    :param decoder_target_data:
    :param length:
    :param mfcc_features:
    :param target_length:
    :param latent_dim:
    :param batch_size:
    :param epochs:
    :return:
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features))
    encoder_states = get_encoder_states(mfcc_features=mfcc_features,
                                        encoder_inputs=encoder_inputs,
                                        latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length))
    decoder_outputs, (decoder_lstm1_layer, decoder_lstm2_layer) = get_decoder_outputs(target_length=target_length,
                                                                                      encoder_states=encoder_states,
                                                                                      decoder_inputs=decoder_inputs,
                                                                                      latent_dim=latent_dim)

    # Dense Output Layers
    decoder_dense = TimeDistributed(Dense(target_length, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Training model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)

    # Inference part

    # Creating encoder model
    encoder_model = Model(encoder_inputs, encoder_states)

    # Input shapes for 1st LSTM layer
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm1 = decoder_lstm1_layer(decoder_inputs, initial_state=decoder_states_inputs)

    # Outputs and states from final LSTM Layer
    decoder_outputs, state_h, state_c = decoder_lstm2_layer(decoder_lstm1)
    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


def train_attention_seq2seq_model(mfcc_features=40, target_length=40, latent_dim=512):
    return 1


def train_cnn_attention_seq2seq_model(encoder_input_data, decoder_input_data, decoder_target_data,
                                      length, mfcc_features=40, target_length=40, latent_dim=512,
                                      batch_size=64, epochs=70):
    """

    :param encoder_input_data:
    :param decoder_input_data:
    :param decoder_target_data:
    :param length:
    :param mfcc_features:
    :param target_length:
    :param latent_dim:
    :param batch_size:
    :param epochs:
    :return:
    """
    cnn_input_shape = (length, mfcc_features)
    # getting CNN model
    cnn_inputs = Input(shape=cnn_input_shape)
    cnn_model = get_cnn_model(cnn_input_shape)
    # Preparing Input shape for LSTM layer from CNN model

    encoder_inputs = cnn_model(cnn_inputs)
    encoder_states = get_encoder_states(mfcc_features=cnn_model.output_shape[2],
                                        encoder_inputs=encoder_inputs,
                                        latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length))
    decoder_outputs = get_decoder_outputs(target_length=target_length,
                                          encoder_states=encoder_states,
                                          decoder_inputs=decoder_inputs,
                                          latent_dim=latent_dim)

    # Dense Output Layers
    decoder_dense = TimeDistributed(Dense(target_length, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Training model

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)

    return model


def train_model(encoder_input_data, decoder_input_data,decoder_target_data,
                latent_dim=512, model_architecture=1):
    """
    Choosing the architecture and running a training
    :param encoder_input_data: Numpy 3dArray
    :param decoder_input_data: Numpy 3dArray
    :param decoder_target_data: Numpy 3dArray
    :param latent_dim: int
    :param epochs: int
    :param batch_size: int
    :param model_architecture: int
    """
    encoder_model, decoder_model = None, None
    length = encoder_input_data.shape[1]
    mfcc_features = encoder_input_data.shape[2]
    target_length = decoder_input_data.shape[2]
    if model_architecture == 1:
        model, encoder_model, decoder_model = train_baseline_seq2seq_model(encoder_input_data=encoder_input_data,
                                                                           decoder_input_data=decoder_input_data,
                                                                           decoder_target_data=decoder_target_data,
                                                                           mfcc_features=mfcc_features,
                                                                           target_length=target_length,
                                                                           latent_dim=latent_dim)
        model_name = "baseline.h5"
    elif model_architecture == 2:
        model = train_attention_seq2seq_model(mfcc_features=mfcc_features,
                                              target_length=target_length,
                                              latent_dim=latent_dim)
        model_name = "attention_based.h5"
    else:
        model = train_cnn_attention_seq2seq_model(encoder_input_data=encoder_input_data,
                                                  decoder_input_data=decoder_input_data,
                                                  decoder_target_data=decoder_target_data,
                                                  length=length,
                                                  mfcc_features=mfcc_features,
                                                  target_length=target_length,
                                                  latent_dim=latent_dim)
        model_name = "cnn_attention.h5"

    model.save(model_name)


    return encoder_model, decoder_model











