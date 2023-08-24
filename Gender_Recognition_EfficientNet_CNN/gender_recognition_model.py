# Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

from glob import glob
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import layers
from keras.callbacks import ModelCheckpoint
from zipfile import ZipFile


# Extracting Zip
with ZipFile('gender.zip') as zipfile:
    zipfile.extractall()


# Hyperparameters and Constants
BATCH_SIZE = 10
EPOCHS = 10
SPLIT = 0.25
IMG_SIZE = 300
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


# Data Preprocessing
X = []
Y = []

data_path = 'faces'

classes = os.listdir(data_path)

for i, name in enumerate(classes):
    images = glob(f'{data_path}/{name}/*.jpg')

    for image in images:
        img = cv2.imread(image)

        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
Y = pd.get_dummies(Y)


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size= SPLIT,
                                                    random_state= 24,
                                                    shuffle= True
                                                    )


# Creating Model Based On EfficientNet
base_model = keras.applications.EfficientNetB3(input_shape= IMG_SHAPE,
                                               include_top= False,
                                               pooling= 'max'
                                               )

model = keras.Sequential([
    base_model,

    layers.Dropout(0.1),

    layers.Dropout(0.2),
    layers.Dense(128, activation= 'relu'),

    layers.Dropout(0.25),
    layers.Dense(2, activation= 'softmax')
])

model.compile(optimizer= 'adam',
              loss= 'binary_crossentropy',
              metrics= ['accuracy']
              )


# Model Callbacks
checkpoint = ModelCheckpoint('output/gender_classification_weights.h5',
                             monitor= 'val_accuracy',
                             save_best_only= True,
                             save_weights_only= True,
                             verbose= 1
                             )


# Model Training
model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          epochs= EPOCHS,
          callbacks= checkpoint,
          validation_data= (X_test, Y_test),
          shuffle= True,
          verbose= 1
          )