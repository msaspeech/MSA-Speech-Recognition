from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers import RepeatVector, TimeDistributed, Dense, Input
from tensorflow.python.keras import Model


MFCC_FEATURES = 40
latent_dim = 1024
# encoder training
encoder_inputs = Input(shape=(None, MFCC_FEATURES))
encoder = LSTM(latent_dim,
               batch_input_shape=(1, None, MFCC_FEATURES),
               stateful=False,
               return_sequences = True,
               return_state = True,
               recurrent_initializer = 'glorot_uniform')

encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c] # 'encoder_outputs' are ignored and only states are kept.

# Decoder training, using 'encoder_states' as initial state.
decoder_inputs = Input(shape=(None, MFCC_FEATURES))

decoder_lstm_1 = LSTM(latent_dim,
                      batch_input_shape = (1, None, MFCC_FEATURES),
                      stateful = False,
                      return_sequences = True,
                      return_state = False,
                      dropout = 0.2,
                      recurrent_dropout = 0.2) # True

decoder_lstm_2 = LSTM(32, # to avoid "kernel run out of time" situation. I used 128.
                     stateful = False,
                     return_sequences = True,
                     return_state = True,
                     dropout = 0.2,
                     recurrent_dropout = 0.2)

decoder_outputs, _, _ = decoder_lstm_2(decoder_lstm_1(decoder_inputs, initial_state = encoder_states))
decoder_dense = TimeDistributed(Dense(MFCC_FEATURES, activation = 'relu'))
decoder_outputs = decoder_dense(decoder_outputs)

# training model
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
training_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
