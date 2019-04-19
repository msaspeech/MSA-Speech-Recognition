from lib import upload_dataset
from models import train_model

(train_encoder_input, train_decoder_input, train_decoder_target), \
(test_encoder_input, test_decoder_input, test_decoder_target) = upload_dataset()

print(train_encoder_input.shape, train_decoder_input.shape, train_decoder_target.shape)


model = train_model(encoder_input_data=train_encoder_input,
                    decoder_input_data=train_decoder_input,
                    decoder_target_data=train_decoder_target,
                    model_architecture=3)
