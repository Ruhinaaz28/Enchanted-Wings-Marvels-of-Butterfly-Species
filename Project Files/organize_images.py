import os
import pandas as pd
import shutil

# Base folder path where 'train' and 'Training_set.csv' exist
base_dir = "D:/Butterfly_Project/butterfly_dataset"
csv_path = os.path.join(base_dir, "Training_set.csv")
images_dir = os.path.join(base_dir, "train")

# Read the CSV file
df = pd.read_csv(csv_path)

# Loop through each image and label
for _, row in df.iterrows():
    filename = row['filename']
    label = row['label']

    # Folder for that class (label)
    label_folder = os.path.join(images_dir, label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    # Move the image file into its class folder
    src = os.path.join(images_dir, filename)
    dst = os.path.join(label_folder, filename)

    if os.path.exists(src):
        shutil.move(src, dst)
    else:
        print(f"❌ File not found: {filename}")

print("✅ All images have been sorted into folders.")
