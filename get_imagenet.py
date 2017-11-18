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

# create the base pre-trained model
base_model = InceptionV3(
    weights='imagenet', include_top=True, input_shape=(139, 139, 3))


#get imagenet dataset
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=[categorical_accuracy, top_k_categorical_accuracy])
eval_data = base_model.evaluate_generator(datagen.flow_from_directory(
       '/root/imagenet/validation',
       target_size=(139, 139),
       batch_size=BATCH_SIZE,
       class_mode='categorical'),
                        steps=100)
for i, metric_name in enumerate(base_model.metrics_names):
   print "   - {0}: {1}".format(metric_name, eval_data[i])
raise
