from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras import backend as K
import PIL
from PIL import Image
import numpy as np

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.optimizers import SGD

edge_size = 139
BATCH_SIZE = 10
NUM_CLASSES = 10

def evaluate_model(model, data):
    x, y = data
    gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    eval_data = model.evaluate_generator(gen.flow(x, y, batch_size=BATCH_SIZE),
                             steps=len(x) / BATCH_SIZE)
    for i, metric_name in enumerate(model.metrics_names):
        print "   - {0}: {1}".format(metric_name, eval_data[i])

def setup_data(data='MNIST'):
    if data=='MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif data=='CIFAR-10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        raise("Invalid data set selected")
    LIMIT = 1500
    (x_train, y_train) = (x_train[:LIMIT], y_train[:LIMIT])
    (x_test, y_test) = (x_test[:LIMIT], y_test[:LIMIT])

    # From https://github.com/fchollet/keras/issues/4465 (changed to use Pillow b.c. had problems with opencv instal on GPU machines)
    x_train = [np.array(PIL.Image.fromarray(i).resize((edge_size, edge_size), Image.ANTIALIAS).convert('RGB')) for i in x_train]
    x_test = [np.array(PIL.Image.fromarray(i).resize((edge_size, edge_size), Image.ANTIALIAS).convert('RGB')) for i in x_test]

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

def fine_tune(base_models, fine_tune_output, data, data_name):
    print "############################################"
    print "### Starting fine tuning for {} dataset".format(data_name)
    # Define the model we'll fine tune
    model = Model(inputs=base_models[0].input, outputs=fine_tune_output)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers:
        layer.trainable = True
    for m in base_models:
        for layer in m.layers:
            layer.trainable = False

    # Compile the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])

    # Set up the fine tune data
    (x_train, y_train), (x_test, y_test) = data
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


    print "x_train shape is {0}, y_train shape is {1}".format(x_train.shape, y_train.shape)

    print "{} performance before training".format(data_name)
    evaluate_model(model, (x_test, y_test))


    print "Stage 1 of {} fine tuning".format(data_name)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=val_datagen.flow(
                            x_test, y_test, batch_size=BATCH_SIZE),
                        validation_steps=len(x_test) / BATCH_SIZE,
                        steps_per_epoch=len(x_train) / BATCH_SIZE,
                        epochs=2, verbose=1)
    print "Stage 1 of {} fine tuning is done.".format(data_name)
    evaluate_model(model, (x_test, y_test))

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=[categorical_accuracy, top_k_categorical_accuracy])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print "Stage 2 of {} fine tuning".format(data_name)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=val_datagen.flow(
                            x_test, y_test, batch_size=BATCH_SIZE),
                        validation_steps=1,
                        steps_per_epoch=len(x_train) / BATCH_SIZE,
                        epochs=2, verbose=1)

    print "Stage 2 of {} fine tuning is done.".format(data_name)
    evaluate_model(model, (x_test, y_test))

    print "############################################"
    print "### End fine tuning for {} dataset".format(data_name)



# create the base pre-trained model
base_model = InceptionV3(
    weights='imagenet', include_top=True, input_shape=(139, 139, 3))

# Setup MNIST output
x = base_model.get_layer('mixed10').output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
mnist_predictions = Dense(NUM_CLASSES, activation='softmax')(x)
mnist_data = setup_data(data='MNIST')

# Setup CIFAR-10 output
x = base_model.get_layer('mixed10').output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
cifar_predictions = Dense(NUM_CLASSES, activation='softmax')(x)
cifar_data = setup_data(data='CIFAR-10')


#print "destination model"
#for i, layer in enumerate(model.layers[-10:]):
#    print(i, layer.name)

#print "source model"
#for i, layer in enumerate(base_model.layers[-10:]):
#    print(i, layer.name)

# Show performances
print "{} performance before fine tuning".format('CIFAR-10')
cifar_model = Model(inputs=base_model.input, outputs=cifar_predictions)
cifar_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])
evaluate_model(cifar_model, cifar_data[1])
print "{} performance before fine tuning".format('MNIST')
mnist_model = Model(inputs=base_model.input, outputs=mnist_predictions)
mnist_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])
evaluate_model(mnist_model, mnist_data[1])

# Fine-tune for CIFAR-10
fine_tune([base_model, mnist_model], cifar_predictions, cifar_data, 'CIFAR-10')

# Show performances again
print "{} performance first after fine tuning".format('CIFAR-10')
cifar_model = Model(inputs=base_model.input, outputs=cifar_predictions)
cifar_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])
evaluate_model(cifar_model, cifar_data[1])
print "{} performance after first fine tuning".format('MNIST')
mnist_model = Model(inputs=base_model.input, outputs=mnist_predictions)
mnist_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])
evaluate_model(mnist_model, mnist_data[1])

# Fine-tune for CIFAR-10
fine_tune([base_model, cifar_model], mnist_predictions, mnist_data, 'MNIST')

# Show performances again
print "{} performance after second fine tuning".format('CIFAR-10')
cifar_model = Model(inputs=base_model.input, outputs=cifar_predictions)
cifar_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])
evaluate_model(cifar_model, cifar_data[1])
print "{} performance after second fine tuning".format('MNIST')
mnist_model = Model(inputs=base_model.input, outputs=mnist_predictions)
mnist_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])
evaluate_model(mnist_model, mnist_data[1])
