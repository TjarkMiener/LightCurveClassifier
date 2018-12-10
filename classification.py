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

def neural_network(x_train,y_train,x_vali,y_vali):
    
    print('Build model...')
    model = Sequential()
            
    model.add(Convolution2D(32, (3, 3),use_bias=False,input_shape=(5,150,150), data_format='channels_first'))
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
    model.add(Dense(14, activation='sigmoid'))
        
    model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
                
    model.summary()
                          
    print('Train...')
    history = model.fit(x_train, y_train, validation_data=(x_vali, y_vali), epochs=15, batch_size=64)
                          
    plt.figure(figsize=(12,8))
    
    plt.subplot(221)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model train vs validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
        
    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
            
    plt.subplot(223)
    resultsTrain = model.predict(x_train)
    plt.scatter(range(871),resultsTrain,c='r')
    plt.scatter(range(871),y_train,c='g')
    #plt.legend(['prediction', 'real'], loc='upper right')
    plt.title('Training prediction vs real')
        
            
    plt.subplot(224)
    resultsVali = model.predict(x_vali)
    plt.scatter(range(109),resultsVali,c='r')
    plt.scatter(range(109),y_vali,c='g')
    #plt.legend(['prediction', 'real'], loc='upper right')
    plt.title('Validation prediction vs real')
        
    plt.subplots_adjust(hspace=0.75)
    plt.suptitle('LCC'.format(eps,steps),fontsize=14)

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
        x_train.append(hdulTrain[i].data)
        redshift_train.append(hdr['PHOTOZ'])
        ra_train.append(hdr['RA'])
        dec_train.append(hdr['DECL'])
        y_train.append(hdr['TARGET'])

    print(collections.Counter(y_train))

    num_vali = 785
    x_vali = []
    redshift_vali = []
    ra_vali = []
    dec_vali = []
    y_vali = []
    
    for i in np.arange(1,num_vali+1,1):
        hdr = hdulVali[i].header
        x_vali.append(hdulVali[i].data)
        redshift_vali.append(hdr['PHOTOZ'])
        ra_vali.append(hdr['RA'])
        dec_vali.append(hdr['DECL'])
        y_vali.append(hdr['TARGET'])

    print(collections.Counter(y_vali))

    neural_network(x_train,y_train,x_vali,y_vali)

    hdulVali.close()
    hdulTrain.close()

#Execute the main function
if __name__ == "__main__":
    main()
