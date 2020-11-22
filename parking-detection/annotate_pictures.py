import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import random
from pathlib import Path

import aux

IN = 'data/cropped'
OUT = 'data/annotated'
aux.mkdir_p(OUT)
file_list = glob.glob(IN + '/*.png')

for fig in file_list:
# fig = random.choice(file_list)

    file_name = Path(fig).stem

    img = cv2.imread(fig)
    cv2.imshow('image', img)
    pressed_key = cv2.waitKey(0)
    cv2.destroyAllWindows()


    if pressed_key == ord('t'):
        folder = 'taken'
    elif pressed_key == ord('f'):
        folder = 'free'
    elif pressed_key == ord('b'):
        folder = 'bad'
    else:
        raise KeyError('Unknown key pressed')

    print(folder)
    destination = f'{OUT}/{folder}'
    aux.mkdir_p(destination)
    cv2.imwrite(f'{destination}/{file_name}.png', img)

