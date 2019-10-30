# load data to tensorflow
import numpy as np # array operations
import matplotlib.pyplot as plt # show the image
import os # iterate through image and join path
import cv2 # do image operation
from tqdm import tqdm
import random # shuffle data
import pickle # save the dataset so we dont have to rebuild the dataset every time
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# data direactories
DATADIR = "train"
VALDDIR = "test" ## test data?

CATEGORIES = ["new_control", "new_mutant"]

# Iterate over the training data set and convert each image into an array
for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # join path the control or mutant dir
    for img in os.listdir(path):  # Iterating over each image
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Convert images to an array and to grayscale
        # grayscale reduce to 1/3 the size of origin image

# Iterate over the test data set and convert each image into an array
for category in CATEGORIES:
    valid_path = os.path.join(VALDDIR, category)
    for img in os.listdir(valid_path):  # Iterating over each image per controls and mutants
        valid_img_array = cv2.imread(os.path.join(valid_path, img), cv2.IMREAD_GRAYSCALE)  # Converting images to array

# resize the cropped images  to 224 x 244px
IMG_SIZE = 224 
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
valid_new_array = cv2.resize(valid_img_array, (IMG_SIZE, IMG_SIZE))

# Creating training dataset
training_data = []
validation_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to control or mutant
        class_num = CATEGORIES.index(category)  # Determine classification (0 => control 1 => mutant) and map to numeric value (converted index)
        for img in tqdm(os.listdir(path)):  # Iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

# Creating test (validation in this case) data in arrays
def create_validation_data():
    for category in CATEGORIES:

        path = os.path.join(VALDDIR, category)
        class_num = CATEGORIES.index(category)  # Determine classification (0 => control 1 => mutant)

        for img in tqdm(os.listdir(path)):  # Iterate over each image
            try:
                valid_img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                valid_new_array = cv2.resize(valid_img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                validation_data.append([valid_new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass


create_training_data()
create_validation_data()

print(len(training_data))

# Randomly shuffle training data for better result
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

# Randomly shuffle validation (test) data
random.shuffle(validation_data)

## Capital X for feature set
X = []
#
# 3 lower y is lables
y = []
Z = []
r = []

for features, label in training_data:
    X.append(features)  # Training data images
    y.append(label)  # Categories

for features, label in validation_data:
    Z.append(features)  # Test data images
    r.append(label)  # Categories

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) ## "1" for grayscale, "3" for colored data
Z = np.array(Z).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# save the Image Dataset created for training/validation using pickle
# Pickling is the process whereby a Python object is converted into a byte stream (Serialization)
pickle_out = open("pickle/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("pickle/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("pickle/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("pickle/y.pickle", "rb")
y = pickle.load(pickle_in)

pickle_out = open("pickle/Z.pickle", "wb")
pickle.dump(Z, pickle_out)
pickle_out.close()

pickle_out = open("pickle/r.pickle", "wb")
pickle.dump(r, pickle_out)
pickle_out.close()

pickle_in = open("pickle/Z.pickle", "rb")
Z = pickle.load(pickle_in)

pickle_in = open("pickle/r.pickle", "rb")
r = pickle.load(pickle_in)
