import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import numpy as np
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
import os
#import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from random import shuffle
#from kerastuner.tuners import RandomSearch
#from kerastuner.engine.hyperparameters import HyperParameters
import time
from classification_models.tfkeras import Classifiers # pip install git+https://github.com/qubvel/classification_models.git


class DataGenerator(keras.utils.Sequence):
    '''
        this class is handled using a lot of function overloading, I believe
        determination of how the generation will parse batches
        essentially, pass the contents of a folder (specifically a list of file names)
        it will then 
        Heavily influenced by https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        by Afshine Amidi and Shervine Amidi
    '''
    def __init__(self, path, file_names, labels, num_classes, dimensions=(224,224), batch_size=16,
                n_channels=1, shuffle=True):
        '''
            just the initialization of each of the variables used in the class
        '''
        # dimensions of the data (1d? 2d? 3? size?)
        self.dimensions = dimensions 
        # size of each of the batches that the NN will parse
        self.batch_size = batch_size 
        # used for figures
        self.labels = labels 
        # used for labeling
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
        loaded_figures = np.empty((self.batch_size, *self.dimensions, self.n_channels))  
        loaded_labels = np.empty((self.batch_size), dtype=int)

        # for each file in the file batch
        for i, file_name in enumerate(shuffled_file_names_batch):
            # load the file (I believe this can be done with pickles, instead using allow_pickle=true in np.load)
            loaded_figure_no_channel = np.load(self.path + file_name)
            # reshape it so that it includes the number of channels (this may need to be reworked)
            # I would instead save the files with channels attached ,instead of loading a file into each channel
            loaded_figures[i,] = np.reshape(loaded_figure_no_channel, (np.shape(loaded_figure_no_channel)[1], np.shape(loaded_figure_no_channel)[0], 1))
            # then insert a label based on the file name without the .npy ( as mine were .npy array file, fastest loading time but larger than pickle)
            loaded_labels[i] = self.labels[int(file_name.replace('.npy', ''))]

        return loaded_figures, loaded_labels

if __name__ == '__main__':

    label_path = 'PP\\train\\'
    figure_path = 'PP\\trainlabel\\'
    train_path = 'PP\\train_check\\'
    test_path = 'PP\\test_check\\'

    

    val_split = 0.7
    test_split = 0.9

    file_names = os.listdir(figure_path)
    shuffle(file_names)
    training_files=file_names[:int(len(file_names)*0.8)]
    validation_files=file_names[int(len(file_names)*0.8):]
    # start time counter to measure how fast it can run
    start = time.perf_counter()
    
    # load in the figures for the neural network
    labels = np.array(np.load(label_path+'my_training_labels.npy', mmap_mode='r', allow_pickle=True), dtype=np.float64)



    training_generator = DataGenerator(path=figure_path, dimensions=(320,240), labels=labels, batch_size=16, file_names=training_files, n_channels=1, num_classes=3, shuffle=True)
    validation_generator = DataGenerator(path=figure_path, dimensions=(320,240), labels=labels, batch_size=16, file_names=validation_files, n_channels=1, num_classes=3, shuffle=True)


    # this segment is used for testing purposes
    # if there is no folder that contains test images, create one and propogate it

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ create model section ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #Instantiate an empty model
    def build_lenet(image_size):
        model = keras.Sequential()

        model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=(image_size[0],image_size[1],1)))
        model.add(Activation('relu'))
        model.add(AveragePooling2D())
        # model.add(BatchNormalization())

        model.add(Conv2D(filters=16, kernel_size=(3, 3)))
        model.add(Activation('relu'))
        model.add(AveragePooling2D())
        # model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dense(units=120))
        model.add(Activation('relu'))
        # model.add(BatchNormalization())

        model.add(Dense(units=84))
        model.add(Activation('relu'))
        # model.add(BatchNormalization())

        model.add(Dense(units=10))
        model.add(Activation('relu'))
        # model.add(BatchNormalization())

        model.add(Dense(units=1))

        return model

    def build_alexnet(image_size, batchNorm = True, maxPool = True):
        model = keras.Sequential()
        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(image_size[0],image_size[1],1), kernel_size=(9,9), strides=(4,4), padding='same', activation='relu'))
        if maxPool:
            model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        else:
            model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))


        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        if maxPool:
            model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        else:
            model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        if batchNorm:
            model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        if batchNorm:
            model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        if batchNorm:
            model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        if maxPool:
            model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        else:
            model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        if batchNorm:
            model.add(BatchNormalization())

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        #model.add(Dense(4096, activation='relu'))
        # Add Dropout to prevent overfitting
        #model.add(Dropout(0.4))
        #if batchNorm:
        #    model.add(BatchNormalization())

        # 2nd Fully Connected Layer
        #model.add(Dense(4096, activation='relu'))
        #model.add(Dropout(0.4))
        #if batchNorm:
        #    model.add(BatchNormalization())

        # 3rd Fully Connected Layer
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.4))
        if batchNorm:
            model.add(BatchNormalization())

        model.add(Dense(128))
        model.add(Activation('relu'))

        # Output Layer
        model.add(Dense(3))
        model.add(Activation('softmax'))

        return model

    def build_resnet(image_size):
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        model = Sequential()
        # conv_base = ResNet50V2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        conv_base = ResNet18(weights='imagenet', input_shape=(image_size[0], image_size[1], 1), pooling='avg')

        # this section is for turning off training of resnet pretrained weights
        # not sure if it's even useful or not
        """
        conv_base.trainable = False
        for layer in conv_base.layers:
            name = layer.__class__.__name__
            if "BatchNormalization" in name:
                layer.trainable = True
        """

        model.add(keras.layers.Input(shape=(image_size[0], image_size[1], 1)))
        model.add(Conv2D(3,(3,3),padding='same'))
        model.add(conv_base)

        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(3, activation='softmax'))



        return model


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ train model section ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # model.summary()

    # Compile the model
    name = '100by200gray_transfer_learn'
    csv_logger = CSVLogger(name + '.csv', append=True, separator=',')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=4, verbose=1)
    mc = ModelCheckpoint(name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    #image_size = []
    #image_size.append()
    #image_size.append(len(train_figures[0][0]))
    # resnet is most optimal, but the pre-constructed network uses 3 channels
    # and not 1
    # model = build_resnet(image_size)
    model = build_alexnet((320,240), maxPool=False, batchNorm=True)
    # model = build_lenet(image_size)
    model.summary()

    # model = build_model(image_size, batchNorm=True)
    # opt = keras.optimizers.SGD(learning_rate=i, momentum=0.9)
    opt = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss='sparse_categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
    model.fit(training_generator,
                        validation_data=validation_generator
                        )

    #test_predictions = model.predict(test_data_wchannel)
    #for i, prediction in enumerate(test_predictions):
    #    print(f'prediction: {prediction} real value: {test_labels[i]}')
    