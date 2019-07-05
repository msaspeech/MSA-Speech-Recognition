from tensorflow.python.keras import models

model = models.load_model("model.h5")
model.summary()
model.save_weights("model_weights.h5")