from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random

app = Flask(__name__)
model = load_model("vgg16_model.h5")

# Auto-sorted class names from training folders
CLASS_NAMES = sorted(os.listdir("train"))

# Fun facts with emojis 🦋
facts = [
    "🦋 Butterflies can see ultraviolet light, invisible to the human eye.",
    "⚡ Some butterflies can fly up to 30 miles per hour!",
    "👣 Butterflies taste with their feet!",
    "🌍 There are around 20,000 species of butterflies worldwide.",
    "☀️ Butterflies can't fly if they're cold—they need sunshine to warm up!",
    "🌸 A butterfly’s life is mostly spent as a caterpillar before it transforms.",
    "🎨 No two butterflies have the same wing pattern!",
    "🧭 Monarch butterflies migrate thousands of miles each year.",
    "🌺 Butterflies help pollinate flowers just like bees!",
    "🦋 The wings of butterflies are covered in tiny scales that reflect light beautifully."
]

@app.route("/")
def welcome():
    return render_template("welcome.html", fact=random.choice(facts))

@app.route("/input")
def input_page():
    return render_template("input.html")

@app.route("/output", methods=["POST"])
def output():
    if 'file' not in request.files:
        return render_template("output.html", prediction=None, image_path=None)
    
    file = request.files['file']
    if file.filename == '':
        return render_template("output.html", prediction=None, image_path=None)
    
    # Save uploaded file in static folder
    filename = "uploaded.jpg"
    path = os.path.join("static", filename)
    file.save(path)

    # Preprocess and predict
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    
    #return render_template('output.html', prediction=prediction, image_path=image_filename)

    return render_template("output.html", prediction=predicted_class, image_path=path)

if __name__ == "__main__":
    app.run(debug=True)
