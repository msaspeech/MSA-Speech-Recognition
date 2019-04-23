import sys
from lib import upload_dataset
from models import train_model, measure_test_accuracy

batch_size = int(sys.argv[0])
epochs = int(sys.argv[1])

(train_encoder_input, train_decoder_input, train_decoder_target), \
(test_encoder_input, test_decoder_input, test_decoder_target) = upload_dataset()

print(train_encoder_input.shape, train_decoder_input.shape, train_decoder_target.shape)


model, encoder_states = train_model(encoder_input_data=train_encoder_input,
                                    decoder_input_data=train_decoder_input,
                                    decoder_target_data=train_decoder_target,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    model_architecture=1)


# accuracy = measure_test_accuracy(test_decoder_input, model, encoder_states, latent_dim=512)
