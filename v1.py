from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
from keras import optimizers
import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt

# dimensions of images.
img_width, img_height = 150, 150

train_data_dir = 'data/trainbread'
validation_data_dir = 'data/validationbread'
test_data_dir = 'data/testbread'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 0
batch_size = 16

# set data format for different backend (Theano/TensorFlow)
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# create model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# loading weights
model.load_weights('third_try.h5')

for layer in model.layers:
    layer.trainable = False

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# testing augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# validation and test augmentation. Only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

pred_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('third_try.h5')

imgs, labels = pred_generator.next()
array_imgs = np.asarray([image.img_to_array(img) for img in imgs])
predictions = model.predict(imgs)
rounded_pred = np.asarray([np.round(i) for i in predictions])

result = [im for im in zip(array_imgs, rounded_pred, labels, predictions)]

plt.figure(figsize=(12, 12))
for ind, val in enumerate(result[:16]):
    plt.subplot(4, 4, ind + 1)
    im = val[0]

    if (int(val[1]) == 1):
        lb = 'кот'
        cl = 'blue'
    if (int(val[1]) == 0):
        lb = 'хлеб'
        cl = 'red'
    plt.axis('off')
    plt.text(60, -8, lb, fontsize=20, color=cl)
    "plt.text(0, -8, val[2], fontsize=12, color='blue')"
    plt.imshow(np.transpose(im, (0, 1, 2)))
plt.show()
