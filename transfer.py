from keras.applications.inception_v3 import InceptionV3
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

edge_size = 139
BATCH_SIZE = 10
NUM_CLASSES = 10
DATA_SET = 'MNIST' # OR 'CIFAR-10'

def evaluate_model(model, x, y):
    gen = ImageDataGenerator()
    eval_data = model.evaluate_generator(gen.flow(x, y, batch_size=BATCH_SIZE),
                             steps=len(x) / BATCH_SIZE)
    for i, metric_name in enumerate(model.metrics_names):
        print "   - {0}: {1}".format(metric_name, eval_data[i])

def setup_data():
    if DATA_SET=='MNIST':
        from keras.datasets import mnist
        (x_train_original, y_train), (x_test_original, y_test) = mnist.load_data()
        # From https://github.com/fchollet/keras/issues/4465
        x_train = [np.array(PIL.Image.fromarray(i).resize((edge_size, edge_size), Image.ANTIALIAS).convert('RGB')) for i in x_train_original]
        x_test = [np.array(PIL.Image.fromarray(i).resize((edge_size, edge_size), Image.ANTIALIAS).convert('RGB')) for i in x_test_original]

        y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    elif DATA_SET=='CIFAR-10':
        from keras.datasets import cifar10
        (x_train_original, y_train), (x_test_original, y_test) = cifar10.load_data()
        # From https://github.com/fchollet/keras/issues/4465 (changed to use Pillow b.c. had problems with opencv instal on GPU machines)
        x_train = [np.array(PIL.Image.fromarray(i).resize((edge_size, edge_size), Image.ANTIALIAS).convert('RGB')) for i in x_train_original]
        x_test = [np.array(PIL.Image.fromarray(i).resize((edge_size, edge_size), Image.ANTIALIAS).convert('RGB')) for i in x_test_original]

        y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    else:
        raise("Invalid data set selected")
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


# create the base pre-trained model
base_model = InceptionV3(
    weights='imagenet', include_top=False, input_shape=(139, 139, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(64, activation='relu')(x)
# and a logistic layer -- let's say we have NUM_CLASSES classes
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[categorical_accuracy, top_k_categorical_accuracy])

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = setup_data()


# Train the model on the new data for a few epochs
datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
LIMIT = 150
(x_train, y_train) = (x_train[:LIMIT, :, :, :], y_train[:LIMIT])
(x_test, y_test) = (x_test[:LIMIT, :, :, :], y_test[:LIMIT])

print "x_train shape is {0}, y_train shape is {1}".format(x_train.shape, y_train.shape)

print "Performance before training"
evaluate_model(model, x_test, y_test)


print "First round of training"
model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=val_datagen.flow(
                        x_test, y_test, batch_size=BATCH_SIZE),
                    validation_steps=len(x_test) / BATCH_SIZE,
                    steps_per_epoch=len(x_train) / BATCH_SIZE,
                    epochs=2, verbose=1)
print "Stage 1 (fine tuning final layers) done."

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
#for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy', metrics=[categorical_accuracy, top_k_categorical_accuracy])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
print "Second round of training"
model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=val_datagen.flow(
                        x_test, y_test, batch_size=BATCH_SIZE),
                    validation_steps=1,
                    steps_per_epoch=len(x_train) / BATCH_SIZE,
                    epochs=2, verbose=1)

print "Performance after training"
evaluate_model(model, x_test, y_test)
