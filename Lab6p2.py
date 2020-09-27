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
import matplotlib.patches as patches
from PIL import Image as im

def onehot(X):
    T = np.zeros((X.shape[0],np.max(X)+1))
    T[np.arange(len(X)),X] = 1 #Set T[i,X[i]] to 1
    return T

def confusion_matrix(Actual,Pred):
    cm=np.zeros((np.max(Actual)+1,np.max(Actual)+1), dtype=np.int)
    for i in range(len(Actual)):
        cm[Actual[i],Pred[i]]+=1
    return cm

def obj_detection(img, model):
    step = 2
    best_sub = None
    best_prob = -np.inf
    best_lc = 0 
    best_tc = 0
    
    for tc in range(0, img.shape[0] - 6, step):
        for lc in range(0, img.shape[1] - 6, step):
            bounded_box = (tc, lc, tc + 7, lc + 7)
            cropped_img = img[bounded_box[0]:bounded_box[2], bounded_box[1]:bounded_box[3]]
            box_prob = np.array(model.predict(cropped_img.reshape(-1, 7, 7, 1)))
            sub_score = box_prob[0][1]
            if best_prob < sub_score:
                best_sub = cropped_img
                best_prob = sub_score
                best_lc = lc
                best_tc = tc
                
    return best_sub, best_prob, best_lc, best_tc
    
mol = np.load('molecules.npy')
no_mol = np.load('no_mol.npy')

X = np.concatenate((mol, no_mol), axis = 0)
y = np.append(np.ones(mol.shape[0]), np.zeros(no_mol.shape[0]), axis = 0)

X = X.reshape(-1,7,7,1)
X = X.astype(float)
y = onehot(y.astype(int))

# split the data in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state=37)

#param
np.set_printoptions(threshold=np.inf) #Print complete arrays
batch_size = 64
epochs = 50
learning_rate = 0.0001
first_layer_filters = 32
second_layer_filters = 64
ks = 2
mp = 2
dense_layer_size = 128

#model
model = Sequential()
model.add(Conv2D(first_layer_filters, kernel_size=(ks, ks),
                 activation='relu',
                 input_shape= (7, 7, 1)))
model.add(MaxPooling2D(pool_size=(mp, mp)))
model.add(Conv2D(second_layer_filters, (ks, ks), activation='relu'))
model.add(MaxPooling2D(pool_size=(mp, mp)))
model.add(Flatten())
model.add(Dense(dense_layer_size, activation='relu'))
model.add(Dense(2, activation='softmax'))

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

im1 = np.array(im.open('mol_imgs/00001.tif'))
im2 = np.array(im.open('mol_imgs/00002.tif'))
im3 = np.array(im.open('mol_imgs/00003.tif'))
im4 = np.array(im.open('mol_imgs/00004.tif'))
im5 = np.array(im.open('mol_imgs/00005.tif'))
im6 = np.array(im.open('mol_imgs/00006.tif'))
im7 = np.array(im.open('mol_imgs/00007.tif'))
im8 = np.array(im.open('mol_imgs/00008.tif'))
im9 = np.array(im.open('mol_imgs/00009.tif'))
ima = np.array(im.open('mol_imgs/00010.tif'))
imb = np.array(im.open('mol_imgs/00011.tif'))
imc = np.array(im.open('mol_imgs/00012.tif'))
imd = np.array(im.open('mol_imgs/00013.tif'))
ime = np.array(im.open('mol_imgs/00014.tif'))
imf = np.array(im.open('mol_imgs/00015.tif'))
im16 = np.array(im.open('mol_imgs/00016.tif'))
im17 = np.array(im.open('mol_imgs/00017.tif'))
im18 = np.array(im.open('mol_imgs/00018.tif'))
im19 = np.array(im.open('mol_imgs/00019.tif'))
im20 = np.array(im.open('mol_imgs/00020.tif'))

sub_box1, sub1_prob = obj_detection(im1, model)
print(sub1_prob)
plt.imshow(sub_box1)
images = [im1, im2, im3, im4, im5, im6, im7, im8, im9, ima, imb, imc, imd, ime, imf, im16, im17, im18, im19, im20]

fig = plt.figure(figsize = (7,7))
row = 5
col = 4

for i in range(1, row*col+1):
    fig.add_subplot(row, col, i)
    sub_box, sub_prob, best_lc = obj_detection(images[i-1], model)
    print(sub_prob)
    plt.imshow(sub_box)
fig.show()

for i in range(20):
    sub_box, sub_prob, lc, tc = obj_detection(images[i], model)
    fig, ax = plt.subplots(1)
    ax.imshow(images[i])
    rect = patches.Rectangle((lc,tc), 7, 7, linewidth=1, edgecolor= 'r', facecolor = 'none')
    ax.add_patch(rect)
    plt.show()

