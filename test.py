from utils import int_to_binary, get_empty_binary_vector, convert_word_to_binary
from tensorflow.python.keras import models
from lib import AudioInput
import numpy as np

transcript = "SOS_ lr}Asp Aljmhwryp fy syAq AlAntxAbAt Alr}Asyp Almqblp lmA*A h*A AlqrAr fy Alwqt Al*y _EOS"
model = models.load_model("trained_models/architecture1.h5")
sample = AudioInput("test.wav", "")
audio = [sample.mfcc.transpose()]
audio_sequence = np.array(audio)

output = model.predict([audio, transcript])
print(output)
