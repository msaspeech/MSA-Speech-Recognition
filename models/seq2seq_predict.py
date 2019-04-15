
def decode_audio_sequence(audio_sequence, encoder_model, decoder_model):
    states_values = encoder_model.predict(audio_sequence)
