import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import DBM


# from tensorflow.keras import layers
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

db = DBM.DB_man()
x_train, y_train, x_test, y_test = db.load()

x_train = [[[1-x_train[i][j][k][0] for k in range(64)] for j in range(48)] for i in range(len(x_train))]
x_test = [[[1-x_test[i][j][k][0] for k in range(64)] for j in range(48)] for i in range(len(x_test))]

x_train = np.array(x_train)
x_test = np.array(x_test)
# reshape the datat for a one channel
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

print (x_train.shape)
print (x_test.shape)

latent_dim = 512

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
      layers.LSTM(64, input_shape=(10, self.latent_dim)) # accepts a series of time steps and the info size input_shape(time, features)
    ])

  def call(self, x):
    lstm = self.lstm(x)
    return lstm
  

autoencoder = Autoencoder(latent_dim)
optimizer = optimizers.Adam(lr=0.0003)
autoencoder.compile(optimizer=optimizer, loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=250,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()

sequence_len = 10
# information is sliced along the time scale 
# ae: hand moves for 15 frames, lstm accepts 10
# it will take frames 1-10, then 2-11, then 3-12, 4-13, 5-14, 6-15
# each one of these needs a particular label
# suggestion: for static images, (a, b, c), the label will be constant
# wheras for the letters j and z, the label will the null (27th case) for the first couple frames
# this way it will only recognize the letter when there is moement

# data currently provided is literally garbage
encoded_imgs_reshape = np.array([encoded_imgs[i:i+sequence_len] for i, _ in enumerate(encoded_imgs) if i+sequence_len < len(encoded_imgs) ])
print(np.shape(encoded_imgs))

lstm = LSTM_model(latent_dim)
optimizer = optimizers.Adam(lr=0.0006)
lstm.compile(optimizer=optimizer, loss=losses.MeanSquaredError())

lstm.fit(encoded_imgs_reshape, y_test[0:-sequence_len],
                epochs=160)


decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
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
