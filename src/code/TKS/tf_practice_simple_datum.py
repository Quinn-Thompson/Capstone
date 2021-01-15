import tensorflow as tf
import numpy as np
import cv2
import DBM


# from tensorflow.keras import layers


db = DBM.DB_man()
x_train, y_train, x_test, y_test = db.load()

x_train = [[[[1-x_train[i][j][k][l] for l in range(3)] for k in range(64)] for j in range(48)] for i in range(len(x_train))]
x_test = [[[[1-x_test[i][j][k][l] for l in range(3)] for k in range(64)] for j in range(48)] for i in range(len(x_test))]

x_train = np.array(x_train)
x_test = np.array(x_test)

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


#build the model by stacking multiple keras layers
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(48, 64, 3)),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(26)
])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# fit the model to minimize the loss
model.fit(x_train, y_train,
          epochs=128,
          shuffle=True,
          validation_data=(x_test, y_test))

model.evaluate(x_test,  y_test, verbose=2)

predictions = model.predict(x_test)

