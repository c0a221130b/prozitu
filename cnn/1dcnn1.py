import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D

epochs = 100

samples = np.loadtxt('mouse_x.csv', delimiter=',')
samples_label = samples[:, 0].astype(int)
samples_data = samples[:, 1:65]

print(samples_label.shape)
print(samples_data.shape)

samples_label = np.reshape(samples_label, (-1, 1, 1))
samples_data = np.reshape(samples_data, (-1, 64, 1))
print(samples_label.shape)
print(samples_data.shape)

model = Sequential()
model.add(Conv1D(64, 8, padding='same', input_shape=(64, 1), activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Conv1D(64, 8, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Conv1D(32, 8, padding='same', activation='relu'))
model.add(Conv1D(1, 8, padding='same', activation='tanh'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(samples_data, samples_label, validation_split=0.1, epochs=epochs)
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend() 
plt.show()