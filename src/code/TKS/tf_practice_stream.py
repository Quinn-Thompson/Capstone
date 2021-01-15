import tensorflow as tf
import numpy as np
import cv2
import DBM
import preproc as PP
import RSC_Wrapper as RSCW


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
  tf.keras.layers.Dense(128)
])


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# fit the model to minimize the loss
model.fit(x_train, y_train,
          epochs=32,
          shuffle=True,
          validation_data=(x_test, y_test))

#model.evaluate(x_test,  y_test, verbose=2)

#predictions = model.predict(x_test)

# The camera object
cam = RSCW.RSC()
# cam.start_camera()

# The database object
dbm = DBM.DB_man()

# the preproccessing object
pp = PP.PreProc()

stream = []
stream.append(np.zeros((48,64), dtype=float))
stream.append(np.zeros((48,64), dtype=float))
stream.append(np.zeros((48,64), dtype=float))
# main collection loop, everything collection-related will take place in
# this loop
while(1):
    # capture the image
    image = cam.capture()

    # proccess the image
    image = pp.preproccess(image)

    # pop and append
    stream[0] = stream[1]
    stream[1] = stream[2]
    stream[2] = 1-image

    # display the image
    frame = np.dstack((stream[0], stream[1], stream[2]))
    print(np.shape(frame))
    cv2.imshow("Depth Veiw", frame)

    frame = np.expand_dims(frame, 0)

    guess = model.predict(frame)
    print(np.argmax(guess))

    # if a key is pressed, start the collection, otherwise loop
    k = cv2.waitKey(333)

    # check to see if we want to leave
    # ESC == 27 in ascii
    if k == 27:
        break