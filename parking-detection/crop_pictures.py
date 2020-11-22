import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import random
from pathlib import Path

raw_path = 'foto_parcheggio_raw'
file_list = glob.glob(raw_path + '/*.JPG')

p1 = (1680, 2119)
p2 = (2335, 2500)





def show_one():
    img = cv2.imread(random.choice(file_list))
    cropped = img[p1[1]:p2[1], p1[0]:p2[0], :]

    fig, ax = plt.subplots(2)
    imshow(img, ax=ax[0])
    imshow(cropped, ax=ax[1])
    plt.show()


def crop_all(destination='data/cropped', only_few=None):
    if only_few is not None:
        fig_list = random.choices(file_list, k=only_few)
    else:
        fig_list = file_list
    for fig in fig_list:
        img = cv2.imread(fig)
        cropped = img[p1[1]:p2[1], p1[0]:p2[0], :]
        file_name = Path(fig).stem
        cv2.imwrite(f'{destination}/{file_name}.png', cropped)


if __name__ == "__main__":
    show_one()
    # crop_all()




