import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import random
from pathlib import Path
import os
import shutil

import myaux as aux

ANNOTATED = 'data/annotated'

TRAIN = 'data/train_and_test/train'
TEST = 'data/train_and_test/test'

classes = ['free', 'taken']

amount_for_test = 3

for cl in classes:
    this_annotated = f'{ANNOTATED}/{cl}'
    this_train = f'{TRAIN}/{cl}'
    this_test = f'{TEST}/{cl}'
    aux.mkdir_p(this_train)
    aux.mkdir_p(this_test)

    file_list = glob.glob(this_annotated + '/*.png')

    random.shuffle(file_list)

    train_list = file_list[amount_for_test:]
    test_list = file_list[:amount_for_test]

    for file in train_list:
        shutil.copy(file, this_train)
    for file in test_list:
        shutil.copy(file, this_test)




