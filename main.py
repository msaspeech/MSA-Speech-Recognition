from lib import generate_decoder_input_target
from utils import upload_data_after_padding

input, target = generate_decoder_input_target()
data = upload_data_after_padding()
for d in data:
    print(d.mfcc.shape)
