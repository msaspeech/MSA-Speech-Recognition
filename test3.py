from tensorflow.python.keras import models, Input, Model
from tensorflow.python.keras.layers import Dropout, Dense, GRU
from etc import settings


def get_multi_output_dense(decoder_outputs, target_length):
    dense_layers = []

    for i in range(0, 16):
        decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense" + str(i))
        new_decoder_output = decoder_dense(decoder_outputs)
        dense_layers.append(new_decoder_output)
    return dense_layers


def get_encoder_states(input_shape, encoder_inputs, latent_dim, return_sequences=False):
    encoder = GRU(latent_dim,
                  input_shape=(None, input_shape),
                  stateful=False,
                  return_sequences=return_sequences,
                  return_state=True,
                  kernel_constraint=None,
                  kernel_regularizer=None,
                  reset_after=True,
                  recurrent_initializer='glorot_uniform',
                  name="encoder_gru_layer")
    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, state_h = encoder(encoder_inputs)
    encoder_states = [state_h]
    if return_sequences:
        return encoder_outputs, encoder_states
    else:
        return encoder_states


def get_decoder_outputs(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_gru1_layer = GRU(latent_dim,
                             input_shape=(None, target_length),
                             return_sequences=True,
                             return_state=True,
                             reset_after=True,
                             kernel_constraint=None,
                             kernel_regularizer=None,
                             name="decoder_gru1_layer")
    decoder_gru1, state_h = decoder_gru1_layer(decoder_inputs, initial_state=encoder_states)

    # Second LSTM Layer
    decoder_gru2_layer = GRU(latent_dim,
                             stateful=False,
                             return_sequences=True,
                             return_state=False,
                             reset_after=True,
                             kernel_constraint=None,
                             kernel_regularizer=None,
                             name="decoder_gru2_layer")
    decoder_gru2, state_h = decoder_gru2_layer(decoder_gru1, initial_state=state_h)

    decoder_gru3_layer = GRU(latent_dim,
                             stateful=False,
                             return_sequences=True,
                             return_state=False,
                             reset_after=True,
                             kernel_constraint=None,
                             kernel_regularizer=None,
                             name="decoder_gru3_layer")
    decoder_outputs, state_h = decoder_gru3_layer(decoder_gru2, initial_state=state_h)

    return decoder_outputs


model = models.load_model("model_word.h5")
model.summary()
model.save_weights("model_word_weights.h5")

encoder_inputs = Input(shape=(None, 40), name="encoder_input")

# Encoder model
encoder_states = get_encoder_states(input_shape=40,
                                    encoder_inputs=encoder_inputs,
                                    latent_dim=350)

# Decoder training, using 'encoder_states' as initial state.
decoder_inputs = Input(shape=(None, 50), name="decoder_input")

decoder_outputs = get_decoder_outputs(target_length=50,
                                      encoder_states=encoder_states,
                                      decoder_inputs=decoder_inputs,
                                      latent_dim=350)
# Dense Output Layers

target_length = 50
decoder_outputs = get_multi_output_dense(decoder_outputs, target_length)

# Generating Keras Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.load_weights("model_word_weights.h5")
model.save("model_word.h5")
