import os.path
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
from keras import optimizers
import h5py
from keras.models import Model
from keras import applications

# dimensions of images.
img_width, img_height = [150] * 2

train_data_dir = 'data/trainbread'
validation_data_dir = 'data/validationbread'
test_data_dir = 'data/testbread'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 5
batch_size = 16


def create_model():

       # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # add the model on top of the convolutional base
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # set the first 15 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:15]:
        layer.trainable = False

    return model

def train(model, lr=1e-4, image_dir=train_data_dir):
    if type(lr) not in ['float', 'int'] or not 0 <= lr <= 1:
        # Default learning rate
        lr = 0.001

    huge_dir_expression = type(image_dir) is 'str'\
                          and os.path.isdir(
                            os.path.join(image_dir, 'bread'))\
                          and os.path.isdir(
                            os.path.join(image_dir, 'cats'))

    if not huge_dir_expression:
        # Default train data dir
        image_dir = train_data_dir

    optimizer = optimizers.SGD(lr=lr, momentum=0.9)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        image_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('weights2.h5')

    print('training completed --> weights2.h5')

def run_demo_v2():
    model = create_model()

    # loading weights

    if os.path.exists('weights2.h5'):
        model.load_weights('weights2.h5')
    else:
        train(model)

    # validation and test augmentation. Only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    pred_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(150, 150),
        batch_size=100,
        class_mode='binary')

    imgs, labels = pred_generator.next()
    array_imgs = np.asarray(
        [image.img_to_array(img) for img in imgs])
    predictions = model.predict(imgs)
    rounded_pred = np.asarray([np.round(i) for i in predictions])

    result = [im for im in
              zip(array_imgs, rounded_pred, labels, predictions)]

    plt.figure(figsize=(12, 12))
    for ind, val in enumerate(result[:16]):
        plt.subplot(4, 4, ind + 1)
        im = val[0]
        if int(val[1]):
            lb = 'Cat'
            cl = 'blue'
        else:
            lb = 'Bread'
            cl = 'red'
        plt.axis('off')
        plt.text(50, -4, lb, fontsize=20, color=cl)
        plt.imshow(im)
    plt.show()

def run_training_v2(lr, image_dir):
    model = create_model()
    train(model, lr, image_dir)

def recognize_v2(target):
    model = create_model()

    #loading weights
    if os.path.exists('weights2.h5'):
        model.load_weights('weights2.h5')
    else:
        train(model)

    if os.path.isfile(target):
        img = image.load_img(target, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        result = 'cat' if prediction else 'bread'
        raw_img = matplotlib.image.imread(target)
        plt.imshow(raw_img)
        plt.text(0, -2, 'I think this is a {}'.format(result), fontsize=20)
        plt.axis("off")
        plt.show()
    else:
        raise IOError('No such file')