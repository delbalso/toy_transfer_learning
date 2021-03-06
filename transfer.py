from keras.applications.mobilenet import MobileNet, preprocess_input
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

edge_size = 224
BATCH_SIZE = 10
NUM_CLASSES = 10

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def evaluate_on_imagenet(model):
    #get imagenet dataset
    # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    eval_data = model.evaluate_generator(datagen.flow_from_directory(
           '/root/imagenet/validation',
           target_size=(edge_size, edge_size),
           batch_size=BATCH_SIZE,
           class_mode='categorical'),steps=100)
    for i, metric_name in enumerate(base_model.metrics_names):
       print "   - {0}: {1}".format(metric_name, eval_data[i])

def evaluate_model(model, data):
    x, y = data

    eval_data = model.evaluate_generator(datagen.flow(x, y, batch_size=BATCH_SIZE),
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
    #for i, layer in enumerate(model.layers):
    #    print "{0}: {1} - {2}".format(i, layer.name, layer.trainable)
    #model.summary()

    # Set up the fine tune data
    (x_train, y_train), (x_test, y_test) = data

    print "x_train shape is {0}, y_train shape is {1}".format(x_train.shape, y_train.shape)

    print "{} performance before training".format(data_name)
    evaluate_model(model, (x_test, y_test))


    print "Stage 1 of {} fine tuning".format(data_name)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=datagen.flow(
                            x_test, y_test, batch_size=BATCH_SIZE),
                        validation_steps=len(x_test) / BATCH_SIZE,
                        steps_per_epoch=len(x_train) / BATCH_SIZE,
                        epochs=2, verbose=1)
    print "Stage 1 of {} fine tuning is done.".format(data_name)
    evaluate_model(model, (x_test, y_test))

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # Choose to train the top 2 blocks, i.e. we will freeze
    for layer in model.layers[:70]:
        layer.trainable = False
    for layer in model.layers[70:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=[categorical_accuracy, top_k_categorical_accuracy])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print "Stage 2 of {} fine tuning".format(data_name)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=datagen.flow(
                            x_test, y_test, batch_size=BATCH_SIZE),
                        validation_steps=1,
                        steps_per_epoch=len(x_train) / BATCH_SIZE,
                        epochs=2, verbose=1)

    print "Stage 2 of {} fine tuning is done.".format(data_name)
    evaluate_model(model, (x_test, y_test))

    print "############################################"
    print "### End fine tuning for {} dataset".format(data_name)



# create the base pre-trained model
base_model = MobileNet(
    weights='imagenet', include_top=True, input_shape=(224, 224, 3))
base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])


#for i, layer in enumerate(base_model.layers[-10:]):
#    print(i, layer.name)

#raise

# Setup CIFAR-10 output
x = base_model.get_layer('dropout').output
x = GlobalAveragePooling2D()(x)
x = Dense(30, activation='relu')(x)
cifar_predictions = Dense(NUM_CLASSES, activation='softmax')(x)
cifar_data = setup_data(data='CIFAR-10')


#print "destination model"
#for i, layer in enumerate(model.layers[-10:]):
#    print(i, layer.name)

#print "source model"
#for i, layer in enumerate(base_model.layers[-10:]):
#    print(i, layer.name)

# Show performances
cifar_model = Model(inputs=base_model.input, outputs=cifar_predictions)
cifar_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])
print "{} performance first after fine tuning".format('CIFAR-10')
evaluate_model(cifar_model, cifar_data[1])
print "{} performance first after fine tuning".format('imagenet')
evaluate_on_imagenet(base_model)

# Fine-tune for CIFAR-10
fine_tune([base_model], cifar_predictions, cifar_data, 'CIFAR-10')

# Show performances again
cifar_model = Model(inputs=base_model.input, outputs=cifar_predictions)
cifar_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])
base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])
print "{} performance first after fine tuning".format('CIFAR-10')
evaluate_model(cifar_model, cifar_data[1])
print "{} performance first after fine tuning".format('imagenet')
evaluate_on_imagenet(base_model)
