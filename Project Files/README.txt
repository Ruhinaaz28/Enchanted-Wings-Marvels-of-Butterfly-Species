=========================================================
     🦋 Enchanted Wings: Butterfly Classification
=========================================================

This project is a butterfly species classification system built using deep learning and a Flask web interface. It uses transfer learning (VGG16) to predict butterfly species from images.

Model: Trained on 75 species, using VGG16
Tech Stack: Python, TensorFlow, Keras, OpenCV, Flask, HTML/CSS

=========================================================
🔧 SETUP INSTRUCTIONS (One-Time Setup)
=========================================================

STEP 1️: Install Anaconda (if not installed)
---------------------------------------------
Download from: https://www.anaconda.com/products/distribution
Install it on your system.

STEP 2️: Create a Virtual Environment
--------------------------------------
Open **Anaconda Prompt** and run:

    conda create -n butterfly-classifier python=3.9
    conda activate butterfly-classifier

STEP 3️: Install Required Packages
-----------------------------------
While inside the virtual environment, run:

    pip install tensorflow keras flask matplotlib numpy opencv-python pillow

STEP 4️: Run the Web Application
----------------------------------
Navigate to the folder where this project is located. For example:

    cd /d D:\Butterfly_Project\butterfly_dataset

Then run:

    python app.py

You should see something like:

    * Running on http://127.0.0.1:5000/

Open your browser and go to that address.

=========================================================
🧠 HOW TO USE
=========================================================

1. Welcome page appears with animated background and a fun butterfly fact.
2. Click "Upload Image" to go to the input page.
3. Select a butterfly image and click "Predict".
4. The model will analyze the image and show:
    ✅ Predicted species
    🖼️ Uploaded image preview
    🎨 Stylish animated layout

=========================================================
🌐 OFFLINE USAGE
=========================================================

This app is fully offline if:
✔ You have this folder with:
    - `vgg16_model.h5`
    - `static/` folder (backgrounds, styles)
    - `templates/` folder (HTML pages)
✔ All packages are installed in Anaconda
✔ No fonts or images are pulled from online sources

=========================================================
📂 FOLDER STRUCTURE
=========================================================

Butterfly_Project/
└── butterfly_dataset/
    ├── app.py                   ← Flask web app script
    ├── vgg16_model.h5           ← Trained butterfly classification model
    ├── train/                   ← Training images (organized by class folders)
    ├── test/                    ← Test images (optional for predictions)
    ├── static/                  ← Static assets (CSS, images, uploads)
    │   ├── background.jpg       ← Beautiful background image
    │   └── uploads/             ← Stores user-uploaded butterfly images
    ├── templates/               ← HTML pages used by Flask
    │   ├── welcome.html
    │   ├── input.html
    │   └── output.html
    ├── Training_set.csv         ← Used for organizing training data
    ├── Testing_set.csv          ← Used for evaluating/testing
    └── README.txt               ← Setup and usage instructions


=========================================================
✨ ENJOY USING ENCHANTED WINGS 🦋
=========================================================


