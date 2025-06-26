import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model_path = 'vgg16_model.h5'
test_dir = 'test'
class_names = sorted(os.listdir('train'))

model = load_model(model_path)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()

test_path = os.path.join(os.getcwd(), test_dir)
test_images = [f for f in os.listdir(test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

if not test_images:
    print("⚠️ No test images found in the 'test' folder.")
else:
    for img_file in test_images:
        print(f'\n🖼️ Predicting: {img_file}')
        predict_image(os.path.join(test_path, img_file))
