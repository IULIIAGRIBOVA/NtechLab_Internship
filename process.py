

#импорт библиотек

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import zipfile
import csv
import sys
import os


import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_file
import json
from sklearn.model_selection import train_test_split, StratifiedKFold

import PIL
from PIL import ImageOps, ImageFilter

#установка параметров

BATCH_SIZE           = 64 # уменьшаем batch если сеть большая, иначе не влезет в память на GPU
LR                   = 1e-2
CLASS_NUM            = 2  # количество классов в нашей задаче
IMG_SIZE             = 224 # какого размера подаем изображения в сеть
IMG_CHANNELS         = 3   
input_shape          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
PYTHONHASHSEED = 0
folder_path  = sys.argv[1] #путь к папке с фото

#загрузка данных для предсказания
sample_submission = pd.DataFrame(columns = ['Id', 'Categorical'])
for i in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path,i)):
        sample_submission = sample_submission.append({'Id': str(i) , 'Categorical': str(0)}, ignore_index=True)

#создание генератора
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_sub_generator = test_datagen.flow_from_dataframe(
    dataframe=sample_submission,
    directory=folder_path,
    x_col="Id",
    y_col=None,
    shuffle=False,
    class_mode=None,
    seed=RANDOM_SEED,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,)

#создание модели с  использованием transfer-learning
base_model = Xception(weights='imagenet', include_top=False, input_shape = input_shape)
# Устанавливаем новую "голову" (head)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

#выходной слой
predictions = Dense(CLASS_NUM, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adamax(lr=LR), metrics=["accuracy"])

#загрузка весов предобученной модели
model.load_weights("best_model.hdf5")

#предсказание
predictions = model.predict_generator(test_sub_generator, steps=len(test_sub_generator), verbose=1)
predictions = np.argmax(predictions, axis=-1) #multiple categories


#создание файла с предсказанием
label_map = {0: 'female', 1: 'male'}
predictions = [label_map[k] for k in predictions]

filenames_with_dir = test_sub_generator.filenames
submission = pd.DataFrame({'Id':filenames_with_dir, 'Category':predictions}, columns=['Id', 'Category'])
submission = submission.set_index('Id')


dict_submission =  {photo: label[0] for photo, label in submission.iterrows()}
with open('process_results.json', 'w') as f:
    f.write(json.dumps(dict_submission))

print(f'Файл process_results.json успешно сохранен в {os.getcwd()}')



