from pathlib import Path
import matplotlib.pyplot as plt
import cv2


def imshow(image, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))