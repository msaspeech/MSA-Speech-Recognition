from lib import upload_dataset, upload_original_data
from utils import decode_transcript, get_character_set

(train_encoder_input, train_decoder_input, train_decoder_target), \
(test_encoder_input, test_decoder_input, test_decoder_target) = upload_dataset()

print(train_encoder_input.shape, train_decoder_input.shape, train_decoder_target.shape)
print(test_encoder_input.shape, test_decoder_input.shape, test_decoder_target.shape)

