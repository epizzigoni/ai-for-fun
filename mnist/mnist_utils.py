import gzip
import hashlib
import os
import requests
import random
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score


W = 28  # pixels
H = W


def fetch(url):
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    try:
        with open(fp, "rb") as f:
            dat = f.read()
    except FileNotFoundError:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


def load_data():
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, H, W))
    y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, H, W))
    y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, y_train, X_test, y_test


def print_shapes(X_train, y_train, X_test, y_test):
    print('Images train size: ', X_train.shape)
    print('Labels train size: ', y_train.shape)
    print('Images test size: ', X_test.shape)
    print('Labels test size: ', y_test.shape)


def plot_random_grid(X, y, grid_n=4):

    idx = np.random.choice(len(X), grid_n * grid_n)
    grid = np.concatenate(X[idx, :, :].reshape(grid_n, H * grid_n, W), axis=1)

    plt.imshow(grid, cmap='gray_r')
    plt.gca().axis('off')
    plt.title('Images')
    plt.show()

    print('\nLabels:\n', y[idx].reshape(grid_n, grid_n).T)


def print_accuracy(y_test, y_pred):
    for num in range(10):
        this_acc = accuracy_score(y_test[y_test == num],
                                  y_pred[y_test == num])
        print(f'Accuracy {num}: {round(this_acc * 100, 2)} %')

    acc = accuracy_score(y_test, y_pred)
    print(f'\nTotal Accuracy: {acc * 100} %')


def plot_correct_and_wrong(X_test, y_test, y_pred, n_plot=8,
                           figsize=(16, 7), fontsize=14):

    idx_correct = random.sample(list(np.where(y_test == y_pred)[0]), k=n_plot)
    idx_wrong = random.sample(list(np.where(y_test != y_pred)[0]), k=n_plot)

    fig, ax = plt.subplots(2, n_plot // 2, figsize=figsize)
    for i, a in zip(idx_correct, ax.flat):
        a.imshow(X_test[i, :, :], cmap='gray_r')
        a.set_title(f'Real = {y_test[i]}, Predicted = {y_pred[i]}', fontsize=fontsize)
        a.axis('off')
    plt.suptitle('CORRECT EXAMPLES:', fontweight="bold", fontsize=fontsize+2, c='g')

    fig, ax = plt.subplots(2, n_plot // 2, figsize=figsize)
    for i, a in zip(idx_wrong, ax.flat):
        a.imshow(X_test[i, :, :], cmap='gray_r')
        a.set_title(f'Real = {y_test[i]}, Predicted = {y_pred[i]}', fontsize=fontsize)
        a.axis('off')
    plt.suptitle('WRONG EXAMPLES:', fontweight="bold", fontsize=fontsize+2, c='r')

    plt.show()









