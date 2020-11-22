import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import glob
import cv2
import aux

ANNOTATED = 'data/annotated'

TRAIN = 'data/train_and_test/train'
TEST = 'data/train_and_test/test'

classes = ['free', 'taken']

amount_for_test = 3

X, y = [], []

for cl in classes:

    this_train = f'{TRAIN}/{cl}'
    file_list = glob.glob(this_train + '/*.png')
    for file in file_list:
        img = cv2.imread(file, 0)
        X.append(img.flatten())
        if cl == 'free':
            y.append(1)
        else:
            y.append(0)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

fig, ax = plt.subplots(2, 3, figsize=(12, 6))

for i, cl in enumerate(classes):
    this_test = f'{TEST}/{cl}'
    file_list = glob.glob(this_test + '/*.png')
    for j, file in enumerate(file_list):
        img = cv2.imread(file, 0)

        aux.imshow(img, ax=ax[i][j])
        ax[i][j].axis('off')

        prediction = neigh.predict([img.flatten()])
        if prediction[0] == 1:
            title = 'This spot is FREE!'
        else:
            title = 'This spot is TAKEN!'
        ax[i][j].set_title(title)

plt.tight_layout()
plt.show()








