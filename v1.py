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


# dimensions of images.
img_width, img_height = [150] * 2

train_data_dir = 'data/trainbread'
validation_data_dir = 'data/validationbread'
test_data_dir = 'data/testbread'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 10
batch_size = 16


def create_model():
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

    model.save_weights('weights.h5')

    print('training completed --> weights.h5')

def run_demo():
    model = create_model()

    # loading weights

    if os.path.exists('weights.h5'):
        model.load_weights('weights.h5')
    else:
        train(model)

    # validation and test augmentation. Only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    pred_generator = test_datagen.flow_from_directory(
        validation_data_dir,
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

    wrong = [x for x in result if x[1] != x[2]]

    mistake = len(wrong) / len(result)
    accuracy = 100 - mistake*100
    print(len(wrong))
    print(len(result))
    print('Mistake -- {}%'.format(mistake*100))

    plt.figure(figsize=(12, 12))
    plt.figtext(0, 0, '            Точность -- {}%'.format(accuracy), fontsize=20)

    for ind, val in enumerate(result[:16]):
        plt.subplot(4, 4, ind + 1)
        im = val[0]
        if int(val[1]):
            lb = 'Кот'
            cl = 'blue'
        else:
            lb = 'Хлеб'
            cl = 'red'
        plt.axis('off')
        plt.text(50, -4, lb, fontsize=20, color=cl)
        plt.imshow(np.transpose(im, (0, 1, 2)))
    plt.show()

def run_training(lr, image_dir):
    model = create_model()
    train(model, lr, image_dir)

def recognize(target):
    model = create_model()

    #loading weights
    if os.path.exists('weights.h5'):
        model.load_weights('weights.h5')
    else:
        train(model)

    if os.path.isfile(target):
        img = image.load_img(target, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        raw_img = matplotlib.image.imread(target)
        plt.imshow(raw_img)
        if prediction:
            result = 'cat'
            plt.text(0, -20, 'Я не хлеб, я кот! Не ешьте меня!', fontsize=20)
        else:
            result = 'bread'
            plt.text(150, -50, 'Это сладкий хлебушек.', fontsize=20)
            plt.text(150, -10, 'Кушайте на здоровье!', fontsize=20)
        plt.axis("off")
        plt.show()
    else:
        raise IOError('No such file')
