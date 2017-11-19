from keras.applications.mobilenet import MobileNet, preprocess_input
from keras import regularizers
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras import backend as K
from datagenerator import DataGenerator
import PIL
from PIL import Image
import numpy as np

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.optimizers import SGD

edge_size = 224
BATCH_SIZE = 50
NUM_CLASSES = 10

image_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)
vimage_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def generator(X_data, y_data):
  # https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
  samples_per_epoch = X_data.shape[0]
  number_of_batches = samples_per_epoch/BATCH_SIZE
  counter=0

  while 1:

    X_batch = np.array(X_data[BATCH_SIZE*counter:BATCH_SIZE*(counter+1)]).astype('float32')
    y_batch = np.array(y_data[BATCH_SIZE*counter:BATCH_SIZE*(counter+1)]).astype('float32')
    counter += 1
    yield X_batch,y_batch

    #restart counter to yeild data in the next epoch as well
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
    for i, metric_name in enumerate(base_model.metrics_names):
        print "   - {0}: {1}".format(metric_name, eval_data[i])


def evaluate_model(model, data):
    x, y = data

    eval_data = model.evaluate_generator(image_datagen.flow(x, y, batch_size=BATCH_SIZE),use_multiprocessing=False,
                                         steps=len(x) / BATCH_SIZE)

    for i, metric_name in enumerate(model.metrics_names):
        print "   - {0}: {1}".format(metric_name, eval_data[i])

def evaluate_model_non_image(model, data):
    x, y = data
    dg = DataGenerator(dim_x = 1024, num_classes=NUM_CLASSES, batch_size = BATCH_SIZE)
    eval_data = model.evaluate_generator(dg.generate(y, x), steps=len(x) / BATCH_SIZE)
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


def fine_tune(base_models, fine_tune_output, data, data_name):
    # Set up the fine tune data
    (x_train, y_train), (x_test, y_test) = data


    print "############################################"
    print "### Starting fine tuning for {} dataset".format(data_name)

    # Define the model we'll fine tune
    model = Model(inputs=base_models[0].input, outputs=fine_tune_output)
    #for i, layer in enumerate(model.layers):
    #    print "{0}: {1} - {2}".format(i, layer.name, layer.trainable)

    frozen_model = Model(
        inputs=base_models[0].input, outputs=model.get_layer('global_average_pooling2d_2').output)
    frozen_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # Generate inputs to new dense layers
    print "Generating bottleneck features"
    train_frozen_inner_activations = frozen_model.predict_generator(image_datagen.flow(x=x_train, batch_size=BATCH_SIZE, shuffle=False),
                                                        steps=len(x_train) / BATCH_SIZE, use_multiprocessing=False)
    test_frozen_inner_activations = frozen_model.predict_generator(vimage_datagen.flow(x=x_test, batch_size=BATCH_SIZE, shuffle=False),
                                                        steps=len(x_test) / BATCH_SIZE, use_multiprocessing=False)

    # Define and compile a model of only the layers on top
    frozen_features_inputs = Input(shape=test_frozen_inner_activations.shape[1:])
    #x = Flatten()(frozen_features_inputs)
    x = Dropout(0.5)(frozen_features_inputs)
    x = Dense(256, activation='relu', bias_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.01))(x) # add dropout?
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax', bias_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.01))(x)
    top_model = Model(inputs=frozen_features_inputs, outputs=predictions)
    top_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[
                  categorical_accuracy])
    #print "Top_model"
    #for i, layer in enumerate(top_model.layers):
    #    print "{0}: {1} - {2}".format(i, layer.name, layer.trainable)
    #top_model.summary()

    #print "{} performance before training".format(data_name)
    #evaluate_model(top_model, (test_frozen_inner_activations, y_test))

    print "Stage 1 of {} fine tuning".format(data_name)
    train_intermediate_datagen = DataGenerator(dim_x = 1024, num_classes=NUM_CLASSES, batch_size = BATCH_SIZE)
    test_intermediate_datagen = DataGenerator(dim_x = 1024, num_classes=NUM_CLASSES, batch_size = BATCH_SIZE)
    print "Training data performance before training"
    evaluate_model_non_image(top_model, (train_frozen_inner_activations, y_train))
    print "Validation data performance before training"
    evaluate_model_non_image(top_model, (test_frozen_inner_activations, y_test))
    top_model.fit_generator(generator(train_frozen_inner_activations, y_train),
                        validation_data=generator(test_frozen_inner_activations, y_test),
                        validation_steps=len(x_test) / BATCH_SIZE,
                        steps_per_epoch=len(x_train) / BATCH_SIZE,
                        epochs=20, verbose=2,
                        callbacks=[EarlyStopping(verbose=2)])


    print "Training data performance after training"
    evaluate_model_non_image(top_model, (train_frozen_inner_activations, y_train))
    print "Validation data performance after training"
    evaluate_model_non_image(top_model, (test_frozen_inner_activations, y_test))
    del train_frozen_inner_activations
    del test_frozen_inner_activations
    print "Stage 1 of {} fine tuning is done.".format(data_name)
    (x_train, y_train), (x_test, y_test) = setup_data(data='CIFAR-10', limit=1500)
    #print "weights before transfer"
    #print(model.get_layer('dense_1').get_weights())
    #model.summary()
    #top_model.summary()

    # Copy over weights
    model.get_layer('dense_1').set_weights(top_model.get_layer('dense_3').get_weights())
    model.get_layer('dense_2').set_weights(top_model.get_layer('dense_4').get_weights())
    #print "weights after transfer"
    #print(model.get_layer('dense_1').get_weights())
    #print(model.get_layer('dense_2').get_weights())

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[
                  categorical_accuracy])

    print "Full model training accuracy"
    evaluate_model(model, (x_train, y_train))
    print "Full model validation accuracy"
    evaluate_model(model, (x_test, y_test))
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from MobileNet. We will freeze the bottom N layers
    # and train the remaining top layers.

    # Choose to train the top 2 blocks, i.e. we will freeze
    for layer in model.layers[:70]:
        layer.trainable = False
    for layer in model.layers[70:]:
        layer.trainable = True

    for i, layer in enumerate(model.layers):
        if i<70: continue
        print "{0}: {1} - {2}".format(i, layer.name, layer.trainable)

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9),
                  loss='categorical_crossentropy', metrics=[categorical_accuracy])



    print "Full model validation accuracy after compile"
    evaluate_model(model, (x_test, y_test))
    print "Full model train accuracy after compile"
    evaluate_model(model, (x_train, y_train))

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    for i, layer in enumerate(model.layers):
        print "{0}: {1} - {2}".format(i, layer.name, layer.trainable)
    print "Stage 2 of {} fine tuning".format(data_name)
    model.summary()
    model.fit_generator(image_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=vimage_datagen.flow(
                            x_test, y_test, batch_size=BATCH_SIZE),
                        validation_steps=len(x_test) / BATCH_SIZE,
                        steps_per_epoch=len(x_train) / BATCH_SIZE, use_multiprocessing=False,
                        epochs=6, verbose=1)

    print "Stage 2 of {} fine tuning is done.".format(data_name)
    evaluate_model(model, (x_test, y_test))

    print "############################################"
    print "### End fine tuning for {} dataset".format(data_name)

# create the base pre-trained model
base_model = MobileNet(
    weights='imagenet', include_top=True, input_shape=(224, 224, 3))
#base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])

# Setup CIFAR-10 output
x = base_model.get_layer('dropout').output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', bias_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.01))(x) # add dropout?
x = Dropout(0.5)(x)
cifar_predictions = Dense(NUM_CLASSES, activation='softmax', bias_regularizer=regularizers.l2(0.01), kernel_regularizer=regularizers.l2(0.01))(x)
cifar_data = setup_data(data='CIFAR-10', limit=5000)

# print "destination model"
# for i, layer in enumerate(model.layers[-10:]):
#    print(i, layer.name)

# print "source model"
# for i, layer in enumerate(base_model.layers[-10:]):
#    print(i, layer.name)

# Show performances
#cifar_model = Model(inputs=base_model.input, outputs=cifar_predictions)
#cifar_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[categorical_accuracy, top_k_categorical_accuracy])
#print "{} performance before fine tuning".format('CIFAR-10')
#evaluate_model(cifar_model, cifar_data[1])
# print "{} performance first after fine tuning".format('imagenet')
# evaluate_on_imagenet(base_model)

# Fine-tune for CIFAR-10
fine_tune([base_model], cifar_predictions, cifar_data, 'CIFAR-10')

# Show performances again
cifar_model = Model(inputs=base_model.input, outputs=cifar_predictions)
cifar_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[
                    categorical_accuracy, top_k_categorical_accuracy])
base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[
                   categorical_accuracy, top_k_categorical_accuracy])
print "{} performance first after fine tuning".format('CIFAR-10')
evaluate_model(cifar_model, cifar_data[1])
print "{} performance first after fine tuning".format('imagenet')
evaluate_on_imagenet(base_model)
