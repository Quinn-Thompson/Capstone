import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import pickle5 as pickle
import cv2
import random
import preproc as PP
import RSC_Wrapper as RSCW
import pickle5 as pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence as seq
from tensorflow.keras.utils import to_categorical
import DBM

class DataGenerator(seq):
    '''
        this class is handled using a lot of function overloading, I believe
        determination of how the generation will parse batches
        essentially, pass the contents of a folder (specifically a list of file names)
        it will then 
        Heavily influenced by https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        by Afshine Amidi and Shervine Amidi
    '''
    def __init__(self, path, file_names, num_classes, labels= None, dimensions=(224,224), batch_size=16,
                n_channels=1, shuffle=True, use_file_labels=False, use_file_figures=False):
        '''
            just the initialization of each of the variables used in the class
        '''
        # dimensions of the data (1d? 2d? 3? size?)
        self.dimensions = dimensions 
        # size of each of the batches that the NN will parse
        self.batch_size = batch_size 
        self.use_file_labels = use_file_labels
        # used for figures
        self.labels = labels 
        self.use_file_figures = use_file_figures
        # used for figures
        self.file_names = file_names
        # number of channels the NN will use (color? 3d?)
        self.n_channels = n_channels
        # shuffle data
        self.shuffle = shuffle 
        self.path = path
        self.num_classes = num_classes
        
        self.on_epoch_end()
    
    def __len__(self):
        '''
            parsed when the NN queries for the length of the actual batch
        '''
        # number of batches per epoch
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        '''
            parses the index of the epoch to return a set of data back to the NN
        '''
        # sfni = shuffled file names indeces
        # slices the total shuffled file list into a set of indeces that correspond to the current batch
        # this approach is taken for iterator shuffling as it does not manipulate the base data
        sfni_iterated_slice = self.shuffled_file_names_indeces[index*self.batch_size:(index+1)*self.batch_size]
        # this obtains the file names based on the passed batch of indexes
        shuffled_file_names_batch  = [self.file_names[x] for x in sfni_iterated_slice]
        # obtain the figures and the labels
        figures, labels = self.acquire_data(shuffled_file_names_batch)

        return figures, labels

    def on_epoch_end(self):
        '''
            at the end of each epoch and after the definition
        '''
        # arrange a list of indeces (0, 1, 2 ..., n) of the length of the number of files
        self.shuffled_file_names_indeces = np.arange(len(self.file_names))
        if self.shuffle == True:
            # randomly shuffle the indeces
            np.random.shuffle(self.shuffled_file_names_indeces)
    
    def acquire_data(self, shuffled_file_names_batch):
        # loaded array as placeholder for batching the CNN
        if self.n_channels > 0:
          loaded_figures = np.empty((self.batch_size, *self.dimensions, self.n_channels))  
        else:
          loaded_figures = np.empty((self.batch_size, *self.dimensions))  
        if self.use_file_labels:
          loaded_labels = np.empty((self.batch_size, 4), dtype=int)
        else:
          loaded_labels = np.empty((self.batch_size, *self.dimensions, self.n_channels))  

        # for each file in the file batch
        for i, file_name in enumerate(shuffled_file_names_batch):
            # load the file (I believe this can be done with pickles, instead using allow_pickle=true in np.load)
            with open(self.path + file_name, "rb") as fd:
              loaded_figure_no_channel = pickle.load(fd)
            # reshape it so that it includes the number of channels (this may need to be reworked)
            # I would instead save the files with channels attached ,instead of loading a file into each channel
            if self.n_channels > 0:
              loaded_figures[i,] = np.reshape(loaded_figure_no_channel, (np.shape(loaded_figure_no_channel)[0], np.shape(loaded_figure_no_channel)[1], 1))
            else:
              loaded_figures[i,] = loaded_figure_no_channel/np.max(loaded_figure_no_channel)
            # then insert a label based on the file name without the .npy ( as mine were .npy array file, fastest loading time but larger than pickle)
            if self.use_file_labels:
              loaded_labels[i,] = to_categorical(self.labels[int(file_name)], 4)
            else:
              with open(self.path + file_name, "rb") as fd:
                loaded_labels_no_channel = pickle.load(fd)
                loaded_labels[i] = np.reshape(loaded_labels_no_channel, (np.shape(loaded_labels_no_channel)[0], np.shape(loaded_labels_no_channel)[1], 1))

        return loaded_figures, loaded_labels

class Autoencoder(Model):
  def __init__(self, latent_dim):
    """
      Breaks the NN into two seperate portions, the encoder, which compacts the data
      into a database, and the decoder, which unfurls the data and attempts to recreate the image
    """
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(48,64,1)), # input data shape
      layers.Flatten(), # flatten it into 1D neuron layer
      layers.Dropout(0.2), # 20% dropout breaks 20% of the weights in a DNN, this limits locality when it comes to 
      # weights that have been trained for a specific label ( a weight that heavily effects a traditional label and is broken will
      #  have to adjust somewhere else) this limits overfitting
      layers.Dense(latent_dim), # densely connects x neurons with the flat image
    ])
    
    self.decoder = tf.keras.Sequential([
      layers.GaussianNoise(0.03), # introduces gaussian distrubted noise that helps regularize the encoded DNN, along with also spreading
      # info 
      # gaussian noise introduced between activation function to help normalize the values that need to be fired through the activation function.
      layers.Activation('relu'), # rectified linear activation, better than sigmoid in most cases as the activations lack snapping 
      # between 0 and 1, whereas relu introduces linearity for stochastic gradiant descent (ae, larger values for activation
      # react accordingly, and negatives will not, which accomplishes the purpose of non-linearity like a sigmoid or tanh). 
      layers.BatchNormalization(), # normalize values between layers so that extremes do not cause rampant exponential increase in weights.
      # this is a more likely case when the activations are monotonic like relu
      # The normalization limits weight increasing and exact weight fitting
      layers.Dropout(0.2),
      layers.Dense(3072), # densely connected layer that is the multiple of the image
      layers.Activation('relu'),
      layers.Reshape((48,64)),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    print('ran')
    decoded = self.decoder(encoded)
    return decoded
    
class LSTM_model(Model):
  def __init__(self, latent_dim):
    super(LSTM_model, self).__init__()
    self.latent_dim = latent_dim   
    self.lstm = tf.keras.Sequential([
      layers.Dropout(0.2),
      layers.LSTM(64, activation='relu', input_shape=(sequence_len, self.latent_dim), return_sequences=True), # accepts a series of time steps and the info size input_shape(time, features)
      layers.LSTM(32, activation='relu', return_sequences=True), # accepts a series of time steps and the info size input_shape(time, features)
      layers.LSTM(32, activation='relu',), # accepts a series of time steps and the info size input_shape(time, features)
      #layers.Dense(512, activation='relu'),
      layers.Dense(latent_dim, activation='relu'),
      layers.Dense(4, activation='softmax')
    ])

  def call(self, x):
    lstm = self.lstm(x)
    return lstm
    

if __name__ == '__main__':

  # from tensorflow.keras import layers
  if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
  else:
    print("Please install GPU version of TF")


  label_path = 'database\\temporallabel\\'
  figure_path = 'database\\temporal\\'
  figure_encode_path = 'database\\temporalencoded\\'

  val_split = 0.7
  test_split = 0.9

  file_names = os.listdir(figure_path)
  file_array_shuffle = random.sample( file_names, len(file_names) )
  training_files=file_names[:int(len(file_array_shuffle)*0.8)]
  validation_files=file_names[int(len(file_array_shuffle)*0.8):]

  
  # load in the figures for the neural network
  with open(label_path + 'temporallabel', "rb") as fd:
    labels = np.array(pickle.load(fd), dtype=np.float64)
  
  unique, counts = np.unique(labels, return_counts=True)
  class_weights = dict(zip(unique, counts))
  
  class_weights[0.0] = (1/class_weights[0.0])*len(labels)
  class_weights[1.0] = (1/class_weights[1.0])*len(labels)
  class_weights[2.0] = (1/class_weights[2.0])*len(labels)
  class_weights[3.0] = (1/class_weights[3.0])*len(labels)*1.1
  
  print(class_weights)


  # create a generator for both the validation and the training
  # this methodoligy causes the training to become noticibly slower
  # as it is now forced to load the file from the hard drive
  # albiet, with chunk loading, the batch from the one folder should all be loaded into memory at once

  # is this methodology even worth it? We will see usage of too much memory after ~300,000 figures,
  # which may seem like a lot, but in reality I have hit this number several times in pervious NN architectures
  # (Not only this, but the previous architexture only included x axis shifting for more figures, so this)
  # (number could easily be hit)
  # is it worth it to set it up for when we are so early on and not even using that many figures yet?
  # Eh, Probably not. ¯\_(ツ)_/¯
  latent_dim = 512
  optimizer = optimizers.Adam(lr=0.0003)
  sequence_len = 12

  autoencoder = Autoencoder(latent_dim)

  if len(os.listdir(figure_encode_path)) == 0:
    training_generator = DataGenerator(path=figure_path, dimensions=(48,64), labels=labels, batch_size=16, file_names=training_files, n_channels=1, num_classes=3, shuffle=True, use_file_labels=False)
    validation_generator = DataGenerator(path=figure_path, dimensions=(48,64), labels=labels, batch_size=16, file_names=validation_files, n_channels=1, num_classes=3, shuffle=True, use_file_labels=False)


    
    autoencoder.compile(optimizer=optimizer, loss=losses.MeanSquaredError(), metrics=['MeanAbsoluteError', 'MeanAbsolutePercentageError'])

    autoencoder.fit(training_generator,
                    epochs=80,
                    validation_data=validation_generator)
    
    file_slice = np.zeros(shape=(16, 48, 64))
    all_encoded_iamges = np.zeros(shape=(len(file_names), latent_dim))
    sequence_array = np.zeros(shape=(sequence_len, latent_dim))
    for i, figure in enumerate(file_names):
      with open(figure_path + figure, "rb") as fd:
        sequence_array = np.roll(sequence_array, shift=-1, axis=0)
        loaded_file = pickle.load(fd)
        #flattened_fle = loaded_file.flatten()
        loaded_file_channel = np.reshape(loaded_file, (1, np.shape(loaded_file)[0], np.shape(loaded_file)[1], 1))
        all_encoded_iamges[i] = autoencoder.encoder(loaded_file_channel).numpy()
        all_encoded_iamges[i] = (all_encoded_iamges[i]-np.min(all_encoded_iamges[i]))/(np.max(all_encoded_iamges[i])-np.min(all_encoded_iamges[i]))
        sequence_array[sequence_len-1] = all_encoded_iamges[i]
        
      if i > sequence_len:
        with open(figure_encode_path + str(i-sequence_len).zfill(6), "bx") as fd:
          pickle.dump(sequence_array, fd)




  


  # information is sliced along the time scale 
  # ae: hand moves for 15 frames, lstm accepts 10
  # it will take frames 1-10, then 2-11, then 3-12, 4-13, 5-14, 6-15
  # each one of these needs a particular label
  # suggestion: for static images, (a, b, c), the label will be constant
  # wheras for the letters j and z, the label will the null (27th case) for the first couple frames
  # this way it will only recognize the letter when there is moement

  # data currently provided is literally garbage
  
  #encoded_imgs_reshape = np.array([encoded_imgs[i:i+sequence_len] for i, _ in enumerate(encoded_imgs) if i+sequence_len < len(encoded_imgs) ])
  #print(np.shape(encoded_imgs))

  file_names = os.listdir(figure_encode_path)
  file_array_shuffle = random.sample( file_names, len(file_names) )
  training_files=file_names[:int(len(file_array_shuffle)*0.8)]
  validation_files=file_names[int(len(file_array_shuffle)*0.8):]
  """all_images_vali_view = np.zeros(shape=(len(os.listdir(figure_path)), 48, 64))
  all_images_vali = np.zeros(shape=(len(file_names), sequence_len, latent_dim))
  for i, figure in enumerate(os.listdir(figure_path)):
    with open(figure_path + figure, "rb") as fd:
      loaded_file = pickle.load(fd)
      all_images_vali_view[i] = np.array(loaded_file)
  for i, figure in enumerate(os.listdir(figure_encode_path)):
    with open(figure_encode_path + figure, "rb") as fd:
      loaded_file = pickle.load(fd)
      all_images_vali[i] = np.array(loaded_file)
  """

  class_weight={}

  training_generator = DataGenerator(path=figure_encode_path, dimensions=(sequence_len,latent_dim), labels=labels[sequence_len:], batch_size=16, file_names=training_files, n_channels=0, num_classes=3, shuffle=True, use_file_labels=True)
  validation_generator = DataGenerator(path=figure_encode_path, dimensions=(sequence_len,latent_dim), labels=labels[sequence_len:], batch_size=16, file_names=validation_files, n_channels=0, num_classes=3, shuffle=True, use_file_labels=True)

  lstm = LSTM_model(latent_dim)
  optimizer = optimizers.Adam(lr=0.00006)
  lstm.compile(optimizer=optimizer, loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

  lstm.fit(training_generator,
                  epochs=130,
                  validation_data=validation_generator,
                  class_weight=class_weights)

  #predictions = lstm.predict(all_images_vali)
  #for i, pred in enumerate(predictions):
    #cv2.imshow("Depth Veiw", all_images_vali_view[i+sequence_len])
    #print(str(max( (v, i) for i, v in enumerate(predictions[i]) )[1]) + ' guess ' + str(i))
    #print(str(labels[i+sequence_len]) + ' actual ' + str(i) + '\n')
    #cv2.waitKey(0)

  # The camera object
  cam = RSCW.RSC()
  pp = PP.PreProc()
  image_sequence = np.zeros((1, sequence_len, latent_dim))
  while(1):
      # capture the image
      image = cam.capture()

      # proccess the image
      image = np.array(pp.preproccess(image))
      loaded_file_channel = np.reshape(image, (1, np.shape(image)[0], np.shape(image)[1], 1))
      image_fltn = autoencoder.encoder(loaded_file_channel).numpy()
      image_fltn = (image_fltn-np.min(image_fltn))/(np.max(image_fltn)-np.min(image_fltn))
      image_sequence = np.roll(image_sequence, shift=-1, axis=1)
      image_sequence[0][sequence_len-1] = image_fltn

      # display the image
      cv2.imshow("Depth Veiw", image)
      

      predictions = lstm.predict(image_sequence)
      #cv2.imshow("Depth Veiw", all_images_vali_view[i+sequence_len])
      print(str(max( (v, i) for i, v in enumerate(predictions[0]) )[1]) + ' guess ')

      # if a key is pressed, start the collection, otherwise loop
      k = cv2.waitKey(100)

      # check to see if we want to leave
      # ESC == 27 in ascii
      if k == 27:
          break


  """
  decoded_imgs = autoencoder.decoder(all_encoded_iamges).numpy()
  n = 10
  plt.figure(figsize=(20, 4))
  for i in range(n):
      # display original
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(all_images[i])
      plt.title("original")
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      # display reconstruction
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(decoded_imgs[i])
      plt.title("reconstructed")
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()
  """