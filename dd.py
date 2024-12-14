import os
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('model.h5')

def preprocess_image(image_path, target_size=(45, 45)):
    img = imread(image_path, as_gray=True)
    img_resized = resize(img, target_size, anti_aliasing=True)
    img_flattened = img_resized.flatten() / 255.0
    return img_flattened

def create_label_to_dir_mapping(base_directory):
    label_to_dir = {}
    directories = sorted([d for d in os.listdir(base_directory) if not d.startswith('.')])  # Ignoră fișierele ascunse
    for label, directory in enumerate(directories):
        label_to_dir[label] = directory
    return label_to_dir

base_directory = "C:\\Users\\PC\\Desktop\\extracted_images"
label_to_dir = create_label_to_dir_mapping(base_directory)

def predict_directory_images(test_directory, model, label_to_dir):
    for image_file in os.listdir(test_directory):
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(test_directory, image_file)
        image_data = preprocess_image(image_path)
        image_data = np.expand_dims(image_data, axis=0)

        predictions = model.predict(image_data)
        predicted_label = np.argmax(predictions, axis=1)[0]
        predicted_dir = label_to_dir[predicted_label]

        print(f"Imagine: {image_file} → Numele directorului prezis: {predicted_dir}")

test_directory = "C:\\Users\\PC\\Desktop"  
predict_directory_images(test_directory, model, label_to_dir)
