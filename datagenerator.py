# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
import numpy as np

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 32, batch_size = 32, num_classes=10, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.num_classes = num_classes
      self.batch_size = batch_size
      self.shuffle = shuffle

  def generate(self, labels, data):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(data)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              data_temp = [data[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
              labels_temp = [labels[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
              yield np.array(data_temp[:self.batch_size]), np.array(labels_temp[:self.batch_size])
              # Generate data
              #X, y = self.__data_generation(labels_temp, data_temp)

              #yield X, y

  def __get_exploration_order(self, data):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(data))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, labels, data):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.dim_x))
      y = np.empty((self.batch_size, self.num_classes))

      # Generate data
      for i, datum in enumerate(data):
          # Store volume
          X[i] = data[i]

          # Store class
          y[i] = labels[i]

      return X, y
