import numpy as np
import os 

pwd = os.getcwd()
path = os.path.join(pwd,"extracted_images")
dir_list = os.listdir(path = path)

print(dir_list)