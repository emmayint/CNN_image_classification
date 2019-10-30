## image plotting

import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.utils.multiclass import unique_labels

CATEGORIES = ["new_control","new_mutant"]

pickle_in = open("pickle/r.pickle","rb")
r = pickle.load(pickle_in)
pickle_in = open("pickle/Z.pickle","rb")
Z = pickle.load(pickle_in)

model = tf.keras.models.load_model("model/new_64x3-CNN.model")

data = model.predict(Z)
print(data)

# Generates confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = classes

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # To display the ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
cm_plot_labels = ['new_control', 'new_mutant']

plot_confusion_matrix(r, data.round(), classes=cm_plot_labels,
                      title='Confusion matrix')

# cm = confusion_matrix(r, np.round(data))
# plot_confusion_matrix(Z, data , cm_plot_labels, title='Confusion Matrix')
# plot_confusion_matrix(r, data, classes=cm_plot_labels, normalize=True,
#                       title='Normalized confusion matrix')

plt.show()