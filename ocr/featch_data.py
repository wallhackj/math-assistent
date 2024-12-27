import os
import numpy as np
import gzip
from PIL import Image
from sklearn.model_selection import train_test_split
from tinygrad import Tensor
from test_models_from_tinygrad import fetch_mnist

dataset_path = "/home/vvallhack/Projects/math-assistent/ocr/data"
classes = sorted(os.listdir(dataset_path))
label_mapping = {cls: idx for idx, cls in enumerate(classes)}

def generate_paths() -> tuple[np.ndarray, np.ndarray]:
    image_paths = [] 
    labels = []

    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        if os.path.isdir(cls_path):
            for img_file in os.listdir(cls_path):
                image_paths.append(os.path.join(cls_path, img_file))
                labels.append(label_mapping[cls])

    image_paths = np.array(image_paths)
    labels = np.array(labels, dtype=np.int8)
    return image_paths, labels

def get_symbol_from_prediction(x: int) -> str:
    for cls, idx in label_mapping.items():
        if idx == x:
            return cls
    return None
  
def fetch_dataset(split: float, tensors=False):
    paths, labels = generate_paths()
    parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    data = parse("/home/vvallhack/Projects/math-assistent/ocr/imagini_45x45.gz")[0x10:].reshape((-1, 45*45)).astype(np.float32) 
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, train_size=split, random_state=42)
    if tensors: return Tensor(X_train).reshape(-1, 1, 45, 45), Tensor(Y_train), Tensor(X_test).reshape(-1, 1, 28, 28), Tensor(Y_test)
    else: return X_train, Y_train, X_test, Y_test
    

print(fetch_dataset(.9))
print(fetch_mnist())