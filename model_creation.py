from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow import keras

from keras.metrics import categorical_crossentropy

pickle_in = open("pickle/X.pickle", "rb")
X = pickle.load(pickle_in)
X = np.asarray(X, dtype=np.float32)

pickle_in = open("pickle/y.pickle", "rb")
y = pickle.load(pickle_in)
y = np.asarray(y, dtype=np.float32)

pickle_in = open("pickle/Z.pickle", "rb")
Z = pickle.load(pickle_in)
Z = np.asarray(Z, dtype=np.float32)

pickle_in = open("pickle/r.pickle", "rb")
r = pickle.load(pickle_in)
r = np.asarray(r, dtype=np.float32)

## normalize input values to [0,1) so it's easier for the neuron network to learn
X = X / 255.0  # Normalizing training data, since default size of the array is 256 (0 - 255)
Z = Z / 255.0  # Normalizing test data, since default size of the array is 256 (0 - 255)
##################
## Dense layer (fully connected layer) feeds all outputs from the previous layer to all its neurons, each neuron providing one output to the next layer.
dense_layers = [1]
layer_sizes = [64]
# Convolutional layer consists of a set of “filters”.
conv_layers = [3]

# model = Sequential()
# # model.load_weights('cache/vgg16_weights.h5')
# vgg16_model = keras.applications.vgg16.VGG16()
# for layer in vgg16_model.layers:
#     model.add(layer)
# model.layers.pop()
# model.outputs = [model.layers[-1].output]
# # model.layers[-1].outbound_nodes = []
# model.add(Dense(2, activation='softmax'))
# for layer in model.layers[:10]:
#     layer.trainable = False
# sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# model = vgg_std16_model(224, 224, 1) 
# model.fit(
#         # train_data, test_data,
#         #   batch_size=batch_size,
#         #   nb_epoch=nb_epoch,
#         #   shuffle=True,
#         #   verbose=1,
#         #   validation_data=(X_valid, Y_valid),

#         X, y,
#         batch_size=27, ## how many samples at a time to pass. Try smaller batch if you're getting OOM (out of memory) errors.
#         epochs=9,
#         shuffle=True,
#         verbose=1,
#         # validation_data=(Z, r),
#         # validation_split=0.3, ## Fraction of the training data to be used as validation data. 
#         # callbacks=[tensorboard]
# )   
# predictions_valid = model.predict(Z, batch_size=10, verbose=1)
# score = log_loss(r, predictions_valid)

# val_loss, val_acc = model.evaluate(Z, r)  # evaluate the out of sample data with model
# print(val_loss)  # model's loss (error)
# print(val_acc)  # model's accuracy

# # Generating CNN Model
# for dense_layer in dense_layers:
#     for layer_size in layer_sizes:
#         for conv_layer in conv_layers:
#             ## unique time stamp for each trained model preventing overwritint/appending when forget to change name
#             NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
#             print(NAME)

#             model = Sequential()  # Using Sequential Keras Model

#             ## add first conv layer, specify firstly the number of kernels or filters
#             ## 1 Layer
#             model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:])) 
#             model.add(Activation('relu'))
#             model.add(MaxPooling2D(pool_size=(2, 2)))

#             ## repeat for more conv layers
#             ## conv_layer - 1 = 2 more layers
#             for l in range(conv_layer - 1): # dont need input shape this time
#                 model.add(Conv2D(layer_size, (3, 3)))
#                 # rectified linear unit
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))

#             ## flatten before apply dense layer
#             ## 1 dense layer = 64 layers?
#             model.add(Flatten())
#             for _ in range(dense_layer):
#                 model.add(Dense(layer_size))
#                 model.add(Activation('relu'))
            
#             ## output layer. 
#             model.add(Dense(1))
#             model.add(Activation('sigmoid'))

#             ## the callback object tensorboard with formatted string. 
#             tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

#             ##
#             model.compile(loss='binary_crossentropy',
#                           optimizer='adam',
#                           metrics=['accuracy'],
#                           )
#             ## Trains the model for a fixed number of epochs (iterations on a dataset). 
#             history = model.fit(X, y,
#                                 batch_size=27, ## how many samples at a time to pass. Try smaller batch if you're getting OOM (out of memory) errors.
#                                 epochs=9,
#                                 shuffle=True,

#                                 # validation_data=(Z, r),
#                                 validation_split=0.3, ## Fraction of the training data to be used as validation data. 
#                                 callbacks=[tensorboard]
#                                 )
            
#             # Returns the loss value & metrics values for the model in test mode with !! OUT OF SAMPLE !! data.
#             val_loss, val_acc = model.evaluate(Z, r)  # evaluate the out of sample data with model
#             print(val_loss)  # model's loss (error)
#             print(val_acc)  # model's accuracy

# model.save('model/splitValidation.model')


# ValueError: Error when checking input: expected input_1 to have shape (224, 224, 3) but got array with shape (224, 224, 1)
# When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).
vgg16_model = keras.applications.vgg16.VGG16(input_shape=(224, 224, 1), weights=None)
model=Sequential()
for layer in vgg16_model.layers:
    model.add(layer)

model.layers.pop()

for layer in model.layers[:10]:
    layer.trainable=False

model.add(Dense(2, activation='softmax'))

## Adam optimization function
model.compile(Adam(lr=.0001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, y,
                    batch_size=27, ## how many samples at a time to pass. Try smaller batch if you're getting OOM (out of memory) errors.
                    epochs=9,
                    shuffle=True,

                                # validation_data=(Z, r),
                    validation_split=0.3, ## Fraction of the training data to be used as validation data. 
                    # callbacks=[tensorboard]
                    )
# Retrieve a list of list results on training and test data sets for each training epoch
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# print(acc)
# print(val_acc)

# epochs = range(len(acc))  # Get number of epochs

# Plot training and validation accuracy per epoch
# plt.plot(epochs, acc)
# plt.plot(epochs, val_acc)
# plt.title('Training and validation accuracy')
# plt.show()
# plt.figure()
#
# plt.plot(epochs, loss)
# plt.plot(epochs, val_loss)
# plt.title('Training and validation loss')
# plt.show()
