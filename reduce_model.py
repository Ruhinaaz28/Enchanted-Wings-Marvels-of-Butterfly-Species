from tensorflow import keras
import numpy as np

# Load your existing model
model = keras.models.load_model("vgg16_model.h5")

# Convert model weights to float16 to reduce size
model.set_weights([w.astype(np.float16) for w in model.get_weights()])

# Save the reduced model
model.save("model_reduced.h5")

print("✅ Model saved successfully as model_reduced.h5")
