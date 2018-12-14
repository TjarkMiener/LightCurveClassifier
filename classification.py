#!/usr/bin/python
#
#   classification.py
#
#   Created by Tjark Miener on 24.09.18.
#   Copyright (c) 2018 Tjark Miener. All rights reserved.
#

import numpy as np
import const
from astropy.io import fits
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import sklearn as sk
#from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dropout,Flatten,BatchNormalization,Activation
from keras.layers import Convolution2D,MaxPooling2D,Dense
from keras import optimizers

def hotoneencoding(val):
    arr = np.zeros((14,), dtype=int)
    if val == 90:
        arr[0] = 1
    elif val == 42:
        arr[1] = 1
    elif val == 65:
        arr[2] = 1
    elif val == 16:
        arr[3] = 1
    elif val == 15:
        arr[4] = 1
    elif val == 62:
        arr[5] = 1
    elif val == 88:
        arr[6] = 1
    elif val == 92:
        arr[7] = 1
    elif val == 67:
        arr[8] = 1
    elif val == 95:
        arr[9] = 1
    elif val == 52:
        arr[10] = 1
    elif val == 6:
        arr[11] = 1
    elif val == 64:
        arr[12] = 1
    elif val == 53:
        arr[13] = 1
    else:
        raise ValueError('Sorry! Class {} isn\'t supported.'.format(val))
    return arr

def neural_network(x_train,y_train,x_vali,y_vali):
    
    print('Build model...')
    model = Sequential()
            
    model.add(Convolution2D(32, (3, 3),use_bias=False,input_shape=(4,150,150), data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.15))
    model.add(Convolution2D(32, (3, 3),use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.15))
    model.add(Flatten())
    model.add(Dense(64, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    #model.add(LeakyReLU(alpha=0.03))
    model.add(Dropout(0.25))
    model.add(Dense(14, activation='softmax'))
        
    model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
                
    model.summary()
                          
    print('Train...')
    history = model.fit(x_train, y_train, validation_data=(x_vali, y_vali), epochs=50, batch_size=64)
                          
    plt.figure(figsize=(12,8))
    
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model train vs validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
        
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    
    plt.subplots_adjust(hspace=0.75)
    plt.suptitle('LCC',fontsize=14)

    #Saving the plots.
    plt.savefig('lossacc.png', dpi = 600)
    plt.close()

def main():
    
    print('Loading data...')
    #np.set_printoptions(threshold=np.nan)
    hdulVali = fits.open('test_12345.fits')
    hdulTrain = fits.open('train_12345.fits')
    
    num_train = 7063
    x_train = []
    redshift_train = []
    ra_train = []
    dec_train = []
    y_train = []
    
    for i in np.arange(1,num_train+1,1):
        hdr = hdulTrain[i].header
        channels = []
        channels.append(hdulTrain[i].data[0])
        channels.append(hdulTrain[i].data[1])
        channels.append(hdulTrain[i].data[2])
        channels.append(hdulTrain[i].data[3])
        x_train.append(channels)
        redshift_train.append(hdr['PHOTOZ'])
        ra_train.append(hdr['RA'])
        dec_train.append(hdr['DECL'])
        y_train.append(hotoneencoding(hdr['TARGET']))


    num_vali = 785
    x_vali = []
    redshift_vali = []
    ra_vali = []
    dec_vali = []
    y_vali = []
    
    for i in np.arange(1,num_vali+1,1):
        hdr = hdulVali[i].header
        channels = []
        channels.append(hdulVali[i].data[0])
        channels.append(hdulVali[i].data[1])
        channels.append(hdulVali[i].data[2])
        channels.append(hdulVali[i].data[3])
        x_vali.append(channels)
        redshift_vali.append(hdr['PHOTOZ'])
        ra_vali.append(hdr['RA'])
        dec_vali.append(hdr['DECL'])
        y_vali.append(hotoneencoding(hdr['TARGET']))
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_vali = np.array(x_vali)
    y_vali = np.array(y_vali)
    print(np.array(x_train).shape)
    print(np.array(x_vali).shape)
    #y_train = np.expand_dims(y_train, axis=2)
#y_vali = np.expand_dims(y_vali, axis=2)

    print(np.array(y_train).shape)
    print(np.array(y_vali).shape)
    #x_train = np.reshape(x_train,(7063,1,150,150,1))
    #x_vali = np.reshape(x_vali,(785,1,150,150,1))
    print(np.array(x_train).shape)
    print(np.array(x_vali).shape)
    
    neural_network(x_train,y_train,x_vali,y_vali)

    hdulVali.close()
    hdulTrain.close()

#Execute the main function
if __name__ == "__main__":
    main()
