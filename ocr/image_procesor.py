from os import listdir
import os
import numpy as np
import pickle
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_image_matrix(image_dir):
    try:
        image_grayscale = Image.open(image_dir).convert('L')
        image_np = np.array(image_grayscale) / 255.0
        img_list = image_np.flatten().tolist()
        return img_list
    except Exception as e:
        print(f"Error : {e}")
        return None

def generate_image_batches(dataset_dir, batch_size=100):
    try:
        directory_list = [d for d in listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        if len(directory_list) < 1:
            print("Train Dataset folder is empty or dataset folder contains no images")
            return
        
        for directory in directory_list:
            image_dir = listdir(f"{dataset_dir}/{directory}")
            train_images, test_images = train_test_split(image_dir, test_size=0.1, random_state=42)

            # Process train images in batches
            for i in range(0, len(train_images), batch_size):
                X_train = []
                Y_train = []
                batch = train_images[i:i + batch_size]
                for images in batch:
                    image_vector = get_image_matrix(os.path.join(dataset_dir, directory, images))
                    if image_vector:
                        X_train.append(image_vector)
                        Y_train.append(directory)
                yield X_train, Y_train  # Return a batch

            # Process test images entirely (small size assumed)
            # X_test = []
            # Y_test = []
            # for images in test_images:
            #     image_vector = get_image_matrix(os.path.join(dataset_dir, directory, images))
            #     if image_vector:
            #         X_test.append(image_vector)
            #         Y_test.append(directory)
            # return X_test, Y_test
    except Exception as e:
        print(f"Error : {e}")
        return None, None

def get_test_images_from_directory(dataset_dir):
    X_test, Y_test = [], []
    try:
        directory_list = [d for d in listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

        if len(directory_list) < 1:
            print("Test Dataset folder is empty or dataset folder contains no images")
            return None, None
        
        for directory in directory_list:
            image_dir = listdir(f"{dataset_dir}/{directory}")
            _, test_images = train_test_split(image_dir, test_size=0.1, random_state=42)

            for images in test_images:
                image_vector = get_image_matrix(os.path.join(dataset_dir, directory, images))
                if image_vector:
                    X_test.append(image_vector)
                    Y_test.append(directory)

        return X_test, Y_test
    except Exception as e:
        print(f"Error : {e}")
        return None, None

def train_model():
    train_dataset_dir = "D:\\Users\\Omnissiah\\Documents\\math-assistent\\ocr\\data"
    batch_size = 100
    random_forest_classifier = RandomForestClassifier(n_estimators=50, warm_start=False)
    
    print("Training the model...")
    for X_train_batch, Y_train_batch in generate_image_batches(train_dataset_dir, batch_size=batch_size):
        print(f"Training on batch with {len(X_train_batch)} samples...")
        random_forest_classifier.fit(X_train_batch, Y_train_batch)
    
    # Get test data
    X_test, Y_test = get_test_images_from_directory(train_dataset_dir)
    
    if X_test and Y_test:
        accuracy_score = random_forest_classifier.score(X_test, Y_test)
        print(f"Model Accuracy Score (Test): {accuracy_score}")

        # Save the trained model
        model_dir = "Model"
        os.makedirs(model_dir, exist_ok=True)
        pickle.dump(random_forest_classifier, open(f"{model_dir}/random_forest_classifier.pkl", 'wb'))
        print(f"Model saved in {model_dir}/random_forest_classifier.pkl")
    else:
        print("Error in test data processing.")

train_model()

