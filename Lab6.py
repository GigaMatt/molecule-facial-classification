from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt

def onehot(X):
    T = np.zeros((X.shape[0],np.max(X)+1))
    T[np.arange(len(X)),X] = 1 #Set T[i,X[i]] to 1
    return T

def confusion_matrix(Actual,Pred):
    cm=np.zeros((np.max(Actual)+1,np.max(Actual)+1), dtype=np.int)
    for i in range(len(Actual)):
        cm[Actual[i],Pred[i]]+=1
    return cm

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1 

X_people = people.data[mask]
y_people = people.target[mask]
y_people = onehot(y_people)

# scale the grey-scale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability:
X_people = X_people / 255.
X_people = X_people.reshape(-1,87,65,1)

# split the data in training and test set
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, test_size = .20, stratify=y_people, random_state=37)

#param
np.set_printoptions(threshold=np.inf) #Print complete arrays
batch_size = 64
epochs = 50 #10
learning_rate = 0.0001
first_layer_filters = 32
second_layer_filters = 64
ks = 4
mp = 2
dense_layer_size = 128

#model
model = Sequential()
model.add(Conv2D(first_layer_filters, kernel_size=(ks, ks),
                 activation='relu',
                 input_shape= (87, 65, 1)))
model.add(MaxPooling2D(pool_size=(mp, mp)))
model.add(Conv2D(second_layer_filters, (ks, ks), activation='relu'))
model.add(MaxPooling2D(pool_size=(mp, mp)))
model.add(Flatten())
model.add(Dense(dense_layer_size, activation='relu'))
model.add(Dense(62, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

#Train network
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          verbose=1)

score = model.evaluate(X_test, y_test, verbose=1)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pred=model.predict_classes(X_test)
print(model.summary())

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

X_train_flipped = np.flip(X_train, 3)
X_train2 = np.concatenate((X_train, X_train_flipped), axis = 0)
y_train2 = np.concatenate((y_train, y_train), axis = 0)

#model2
model2 = Sequential()
model2.add(Conv2D(first_layer_filters, kernel_size=(ks, ks),
                 activation='relu',
                 input_shape= (87, 65, 1)))
model2.add(MaxPooling2D(pool_size=(mp, mp)))
model2.add(Conv2D(second_layer_filters, (ks, ks), activation='relu'))
model2.add(MaxPooling2D(pool_size=(mp, mp)))
model2.add(Flatten())
model2.add(Dense(dense_layer_size, activation='relu'))
model2.add(Dense(62, activation='softmax'))

model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

#Train network
history2 = model2.fit(X_train2, y_train2,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          verbose=1)

score2 = model2.evaluate(X_test, y_test, verbose=1)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pred2=model2.predict_classes(X_test)
print(model2.summary())