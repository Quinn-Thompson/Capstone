import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
import DBM


# from tensorflow.keras import layers


db = DBM.DB_man()
x_train, y_train, x_test, y_test = db.load()

x_train = [[[1-x_train[i][j][k][0] for k in range(64)] for j in range(48)] for i in range(len(x_train))]
x_test = [[[1-x_test[i][j][k][0] for k in range(64)] for j in range(48)] for i in range(len(x_test))]

x_train = np.array(x_train)
x_test = np.array(x_test)
"""
(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
"""

"""
# just train on the first 7 letters, for shits and giggles
x_train_t = []
y_train_t = []

for i in range(len(x_train)):
  if y_train[i] >= 0 and y_train[i] <= 6:
    x_train_t.append(x_train[i])
    y_train_t.append(y_train[i])

x_test_t = []
y_test_t = []

for i in range(len(x_test)):
  if y_test[i] >= 0 and y_test[i] <= 6:
    x_test_t.append(x_test[i])
    y_test_t.append(y_test[i])

x_train = np.array(x_train_t)
x_test = np.array(x_test_t)

y_train = np.array(y_train_t)
y_test = np.array(y_test_t)
"""

print (x_train.shape)
print (x_test.shape)

latent_dim = 512

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(48,64)), 
      layers.Conv1D(8, 5, activation='relu', padding='same', strides=2),
      #layers.LocallyConnected1D(16, 5),
      #layers.Conv1D(8, 3, activation='relu', padding='same', strides=2),
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(3072, activation='sigmoid'),
      layers.Reshape((48,64))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  

autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=512,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()

print(np.shape(encoded_imgs))

decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

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

