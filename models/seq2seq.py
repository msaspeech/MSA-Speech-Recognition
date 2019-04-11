from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers import RepeatVector, TimeDistributed, Dense, Input, CuDNNLSTM
from tensorflow.python.keras import Model


MFCC_FEATURES = 40
TARGET_LENGTH = 306
latent_dim = 512
# encoder training

encoder_inputs = Input(shape=(None, ))

encoder = CuDNNLSTM(latent_dim,
                    batch_input_shape=(None, None, MFCC_FEATURES),
                    stateful=False,
                    return_state=True,
                    recurrent_initializer='glorot_uniform')

# 'encoder_outputs' are ignored and only states are kept.
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]


# Decoder training, using 'encoder_states' as initial state.
decoder_inputs = Input(shape=(None, ))

decoder_lstm_1 = CuDNNLSTM(latent_dim,
                           batch_input_shape=(None, None, MFCC_FEATURES),
                           stateful=False,
                           return_state=False,
                           dropout=0.2,
                           recurrent_dropout=0.2)

decoder_lstm_2 = CuDNNLSTM(latent_dim,
                           stateful=False,
                           return_sequences=True,
                           return_state=True,
                           dropout=0.2,
                           recurrent_dropout=0.2)

decoder_outputs, _, _ = decoder_lstm_2(decoder_lstm_1(decoder_inputs, initial_state = encoder_states))
decoder_dense = TimeDistributed(Dense(TARGET_LENGTH, activation='relu'))
decoder_outputs = decoder_dense(decoder_outputs)


# training model
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
training_model.compile(optimizer='adam', loss='mean_squared_error')
