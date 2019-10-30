import cv2
import tensorflow as tf

CATEGORIES = ["new_control", "new_mutant"]

def prepare(filepath):
    IMG_SIZE = 224 ## same number as training image size
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  ## read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) ## resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) ## return the image with shaping that TF wants.
model = tf.keras.models.load_model("model/64x3-CNN.model") ## load in the model

prediction = model.predict([prepare('28986885.jpg')]) ## always pass a list (of file path)
print(prediction)
print(CATEGORIES[int(prediction[0][0])]) ## formatting the output (eg from 0.0 to string "control")