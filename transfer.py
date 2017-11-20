from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras import backend as K
import PIL
from PIL import Image
import numpy as np

from keras.datasets import mnist, cifar10
from keras.optimizers import SGD

import os.path
import pickle

edge_size = 139
BATCH_SIZE = 50
NUM_CLASSES = 10

image_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, horizontal_flip=True)
vimage_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


def generator(X_data, y_data):
  # https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
  samples_per_epoch = X_data.shape[0]
  number_of_batches = samples_per_epoch / BATCH_SIZE
  counter = 0

  while 1:

    X_batch = np.array(
        X_data[BATCH_SIZE * counter:BATCH_SIZE * (counter + 1)]).astype('float32')
    y_batch = np.array(
        y_data[BATCH_SIZE * counter:BATCH_SIZE * (counter + 1)]).astype('float32')
    counter += 1
    yield X_batch, y_batch

    # restart counter to yeild data in the next epoch as well
    if counter == number_of_batches:
        counter = 0

def evaluate_on_imagenet(model):
    # get imagenet dataset
    # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    eval_data = model.evaluate_generator(image_datagen.flow_from_directory(
        '/root/imagenet/validation',
        target_size=(edge_size, edge_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical'), steps=100)
    for i, metric_name in enumerate(model.metrics_names):
        print "   - {0}: {1}".format(metric_name, eval_data[i])


def evaluate_model(model, data):
    x, y = data
    eval_data = model.evaluate_generator(image_datagen.flow(x, y, batch_size=BATCH_SIZE), use_multiprocessing=False,
                                         steps=len(x) / BATCH_SIZE)
    for i, metric_name in enumerate(model.metrics_names):
        print "   - {0}: {1}".format(metric_name, eval_data[i])


def setup_data(data='MNIST', limit=1500):
    if data == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif data == 'CIFAR-10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        raise("Invalid data set selected")
    (x_train, y_train) = (x_train[:limit], y_train[:limit])
    (x_test, y_test) = (x_test[:limit], y_test[:limit])

    # From https://github.com/fchollet/keras/issues/4465 (changed to use Pillow b.c. had problems with opencv instal on GPU machines)
    x_train = [np.array(PIL.Image.fromarray(i).resize(
        (edge_size, edge_size), Image.ANTIALIAS).convert('RGB')) for i in x_train]
    x_test = [np.array(PIL.Image.fromarray(i).resize(
        (edge_size, edge_size), Image.ANTIALIAS).convert('RGB')) for i in x_test]

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


def learn_top_layers(base_model, fine_tune_output, data, data_name):
    # Set up the fine tune data
    (x_train, y_train), (x_test, y_test) = data
    frozen_model = Model(
        inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    frozen_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # Generate inputs (bottleneck features) to new dense layers
    if (os.path.isfile("train_bottlenecks.p") and os.path.isfile("test_bottlenecks.p")):
        print "Loading bottleneck features"
        train_frozen_inner_activations=np.load(
            open("train_bottlenecks.p", "rb"))
        test_frozen_inner_activations=np.load(open("test_bottlenecks.p", "rb"))
    else:
        print "Generating new bottleneck features"
        train_frozen_inner_activations=frozen_model.predict_generator(image_datagen.flow(x=x_train, batch_size=BATCH_SIZE, shuffle=False),
                                                            steps=len(x_train) / BATCH_SIZE)  # , use_multiprocessing=False)
        test_frozen_inner_activations=frozen_model.predict_generator(image_datagen.flow(x=x_test, batch_size=BATCH_SIZE, shuffle=False),
                                                            steps=len(x_test) / BATCH_SIZE)  # , use_multiprocessing=False)

        np.save(open("train_bottlenecks.p", "wb"),
                train_frozen_inner_activations)
        np.save(open("test_bottlenecks.p", "wb"),
                test_frozen_inner_activations)


    # Define top_model, a model that is equivalent to the top layers we'll want
    bottleneck_inputs=Input(shape=test_frozen_inner_activations.shape[1:])
    x=Dropout(0.5)(bottleneck_inputs)
    x=Dense(256, activation='relu', bias_regularizer=regularizers.l2(
        0.001), kernel_regularizer=regularizers.l2(0.001))(x)  # add dropout?
    x=Dropout(0.5)(x)
    predictions=Dense(NUM_CLASSES, activation='softmax', bias_regularizer=regularizers.l2(
        0.01), kernel_regularizer=regularizers.l2(0.01))(x)
    top_model=Model(inputs=bottleneck_inputs, outputs=predictions)
    top_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[
                  categorical_accuracy])

    # Train top_model
    top_model.fit_generator(generator(train_frozen_inner_activations, y_train),
                        validation_data=generator(
                            test_frozen_inner_activations, y_test),
                        validation_steps=len(x_test) / BATCH_SIZE,
                        steps_per_epoch=len(x_train) / BATCH_SIZE,
                        epochs=50, verbose=2,
                        callbacks=[EarlyStopping(verbose=2)])

    # Copy over weights
    model.get_layer('dense_1').set_weights(
        top_model.get_layer('dense_3').get_weights())
    model.get_layer('dense_2').set_weights(
        top_model.get_layer('dense_4').get_weights())

# create the base pre-trained model
base_model=InceptionV3(
    weights='imagenet', include_top=True, input_shape=(edge_size, edge_size, 3))
# base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[categorical_accuracy, top_k_categorical_accuracy])

# Setup CIFAR-10 output
x=base_model.get_layer('avg_pool').output
x=Dropout(0.3)(x)
x=Dense(256, activation='relu', bias_regularizer=regularizers.l2(0.001),
        kernel_regularizer=regularizers.l2(0.001))(x)  # add dropout?
x=Dropout(0.3)(x)
cifar_predictions=Dense(NUM_CLASSES, activation='softmax', bias_regularizer=regularizers.l2(
    0.001), kernel_regularizer=regularizers.l2(0.001))(x)
cifar_data=setup_data(data='CIFAR-10', limit=10000)

# Define the model we'll fine tune
model=Model(inputs=base_model.input, outputs=cifar_predictions)

# Train top layer for CIFAR-10
learn_top_layers(model, cifar_predictions, cifar_data, 'CIFAR-10')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[
              categorical_accuracy])

# Show performances again
print "{} performance first after learning new top layers".format('CIFAR-10')
evaluate_model(model, cifar_data[1])
# print "{} performance first after fine tuning".format('imagenet')
# evaluate_on_imagenet(base_model)
