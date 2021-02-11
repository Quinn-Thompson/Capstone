import numpy as np
import pickle5 as pickle
from os import path
from collections import defaultdict
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import string
from tensorflow.keras import layers
#from skimage import measure

# make the matplotlib plots interactive, this allows them to
# be updated
plt.ion()


class PreProc:
    def __init__(self, blur=False, scale=True, rmbkg=True,
                 resize=True):
        # This preproccessor class is intended to be used for training
        # The basic functionality of this class includes: capturing images,
        # image pre-proccessing, mutating images, and saving as a pickle

        # some pre-proccessing flags to determine the level of pre-proccessing
        # preformed to captured images
        self.f_blur = blur
        self.f_scale = scale
        self.f_rm_background = rmbkg
        self.f_resize = resize

        # create the Keras data augmentation layer
        self.augment = tf.keras.Sequential([

            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.1),
            tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),

        ])

    def preproccess(self, image):
        # Alright, based on the flags, proccess the image stored in self.image

        self.image = image

        # remove the background (must be done before rescaling)
        if self.f_rm_background:
            self.remove_background()

        # rescale the image to be in the range of 0 to 1
        if self.f_scale:
            self.scale()

        # blur the image to reduce noise
        if self.f_blur:
            self.blur()

        if self.f_resize:
            self.resize()

        return self.image

    def remove_background(self):
        # First, find the minimum value in the depth image
        # for the depth image, the closer the object is to the camera,
        #  the smaller
        # the value. Thus, we want to find the closest pixel
        minVal = np.min(self.image[np.nonzero(self.image)])

        # Now we want to mask off the background. We do this by
        # setting any pixel thats further away than 1500 units from
        # the closest pixel to zero
        self.image[self.image > 1500 + minVal] = 0.

    def scale(self):
        # After masking off the background, find the furthest distance
        # in our ROI
        # (note, this could be minVal + 1500, but it could be smaller)
        maxVal = np.max(self.image[np.nonzero(self.image)])

        # Now we preform two operations at once, the first is to scale the ROI
        # relitive to itself
        # as such, the closest pixel should be near zero, and the furthest
        # near one.
        # Second, we raise this value to the fourth power, this is to help
        # make minor differences)
        # between pixels more distinct.
        # For example: the sign 'A' versus the sign 'S'
        # 'A' has the thumb closer to the camera than in 'S', but the
        # difference is thousanths of units. Thus, by raising to the fourth
        # power, we can increase that difference
        self.image = (self.image / maxVal) ** 4

        # Any small value (arbitrarily defined as smaller than 0.001,
        #  typically in the range ~E-5) is the
        # result of floating point errors with the masked off pixels
        # (I beleive)
        # Set these background pizels as 1, the furthest away in our range
        self.image[self.image < 0.001] = 1

    def blur(self):
        # preform a blur on the image to reduce noise
        self.image = ndimage.uniform_filter(self.image, size=10)

    def resize(self, new_shape=(48,64), operation='mean'):
        """
        Bins an ndarray in all axes based on the target shape, by summing or
            averaging.

        Number of output dimensions must match number of input dimensions and
            new axes must divide old ones.

        Example
        -------
        >>> m = np.arange(0,100,1).reshape((10,10))
        >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
        >>> print(n)

        [[ 22  30  38  46  54]
         [102 110 118 126 134]
        [182 190 198 206 214]
        [262 270 278 286 294]
         [342 350 358 366 374]]

        """

        ndarray = self.image

        operation = operation.lower()
        if operation not in ['sum', 'mean']:
            raise ValueError("Operation not supported.")
        if ndarray.ndim != len(new_shape):
            raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
        compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(ndarray, operation)
            ndarray = op(-1*(i+1))

        self.image = ndarray

    def expand_ds_augmentation(self, ds):

        data_augmentation = tf.keras.Sequential([
          layers.experimental.preprocessing.
          RandomFlip("horizontal_and_vertical"),
          layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        print(type(ds))
        dsn = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                     num_parallel_calls=AUTOTUNE)

        print(type(dsn))
        dsn = dsn.prefetch(2)
        print(type(dsn))

        return ds

        ds_new = ds.map(lambda x, y: (self.augment(x, training=True), y))
        return ds.concatonate(ds_new)
