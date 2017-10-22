from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import numpy as np
import pandas as pd
import h5py
import os.path
import matplotlib.pyplot as plt

# path to the model weights files.
weights_path = 'data/weights/vgg16_weights.h5'
top_model_weights_path = 'v5_weights.h5'
# dimensions of images.
img_width, img_height = 150, 150

train_data_dir = 'data/trainbread'
validation_data_dir = 'data/validationbread'
test_data_dir = 'data/testbread'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 10
batch_size = 16

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

#load weights
if (os.path.exists(top_model_weights_path)):
    top_model.load_weights(top_model_weights_path)
else:
    print("Веса не найдены.")

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

pred_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary')

model.summary()

# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=2)

top_model.save_weights(top_model_weights_path)

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
        label = 'кот'
        lb_color = 'blue'
    if (int(val[1]) == 0):
        label = 'хлеб'
        lb_color = 'red'
    plt.axis('off')
    plt.text(60, -8, label, fontsize=20, color=lb_color)
    #plt.text(0, -8, val[2], fontsize=12, color='blue')
    plt.imshow(np.transpose(im, (0, 1, 2)))
plt.show()
