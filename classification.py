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

import sklearn as sk
#from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dropout,Flatten,BatchNormalization,Activation
from keras.layers import Convolution2D,MaxPooling2D,Dense
from keras import optimizers

def createLCwithUL(ts,ef1000,ef1000_ul,num_iter):
    from sklearn import preprocessing
    lightcurves=[]
    for source in num_iter:
        lightcurveEF1000=[]
        for bin in np.arange(0,const.BINNUMBER,1):
            if (ts[source][bin]>4):
                lightcurveEF1000.append([ef1000[source][bin]])
            else:
                lightcurveEF1000.append([ef1000_ul[source][bin]])
        #lightcurveEF1000=np.array(lightcurveEF1000)
        lightcurveEF1000=preprocessing.scale(lightcurveEF1000)
        lightcurves.append(lightcurveEF1000)
    #lightcurves=np.array(lightcurves)
    return lightcurves

def recurrence_plot(s, eps=None, steps=None):
    import sklearn.metrics.pairwise
    if eps==None: eps=0.1
    if steps==None: steps=10
    d = sk.metrics.pairwise.pairwise_distances(s)
    d = np.floor(d / eps)
    d[d > steps] = steps
    #Z = squareform(d)
    return d


def neural_network(eps,steps,x_trainData,y_train,x_valiData,y_vali):
    
    print('(Eps={};Steps={})'.format(eps,steps))
    x_train = []
    for source in np.arange(0,const.NUMBERTRAINSET,1):
        x_train.append([recurrence_plot(x_trainData[source],eps,steps)])
    x_vali = []
    for source in np.arange(0,const.NUMBERVALISET,1):
        x_vali.append([recurrence_plot(x_valiData[source],eps,steps)])
    x_train = np.array(x_train)
    x_vali = np.array(x_vali)
            
    print(x_train.shape)
    print(x_vali.shape)
            
    print('Build model...')
    model = Sequential()
            
    model.add(Convolution2D(32, (3, 3),use_bias=False,input_shape=(1,119,119), data_format='channels_first'))
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
    model.add(Dense(1, activation='sigmoid'))
        
    model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
                
    model.summary()
                          
    print('Train...')
    history = model.fit(x_train, y_train, validation_data=(x_vali, y_vali), epochs=25, batch_size=64)
                          
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
    plt.suptitle('Eps={}; Steps={}'.format(eps,steps),fontsize=14)

    #Saving the plots.
    plt.savefig('lossaccEps{}Steps{}.png'.format(eps,steps), dpi = 600)
    plt.close()

def main():
    
    print('Loading data...')
    #np.set_printoptions(threshold=np.nan)
    hdulTrain = fits.open('training.fits')
    hdulVali = fits.open('validation.fits')
    #hdulTest = fits.open('test.fits')
    #hdulBCU = fits.open('bcu.fits')
    
    dataTrain = hdulTrain[1].data
    dataVali = hdulVali[1].data
    #dataTest = hdulTest[1].data
    #dataBCU = hdulBCU[1].data
    
    indTrain=np.arange(0,const.NUMBERTRAINSET,1)
    indTrainBLL=np.nonzero((dataTrain.field('CLASS')=='bll') | (dataTrain.field('CLASS')=='BLL'))[0]
    indTrainFSRQ=np.nonzero((dataTrain.field('CLASS')=='fsrq') | (dataTrain.field('CLASS')=='FSRQ'))[0]
    
    indVali=np.arange(0,const.NUMBERVALISET,1)
    indValiBLL=np.nonzero((dataVali.field('CLASS')=='bll') | (dataVali.field('CLASS')=='BLL'))[0]
    indValiFSRQ=np.nonzero((dataVali.field('CLASS')=='fsrq') | (dataVali.field('CLASS')=='FSRQ'))[0]
    
    #indTest=np.arange(0,const.NUMBERTESTSET,1)
    #indTestBLL=np.nonzero((dataTest.field('CLASS')=='bll') | (dataTest.field('CLASS')=='BLL'))[0]
    #indTestFSRQ=np.nonzero((dataTest.field('CLASS')=='fsrq') | (dataTest.field('CLASS')=='FSRQ'))[0]
    #indBCU=np.nonzero((dataBCU.field('CLASS')=='bcu'))[0]
    
    tsTrain=dataTrain.field('ts')
    tsVali=dataVali.field('ts')
    #tsTest=dataTest.field('ts')
    #tsBCU=dataBCU.field('ts')
    
    ef1000Train=dataTrain.field('eflux1000')
    ef1000Vali=dataVali.field('eflux1000')
    #ef1000Test=dataTest.field('eflux1000')
    #ef1000BCU=dataBCU.field('eflux1000')
    
    ef1000_ulTrain=dataTrain.field('eflux1000_ul95')
    ef1000_ulVali=dataVali.field('eflux1000_ul95')
    #ef1000_ulTest=dataTest.field('eflux1000_ul95')
    #ef1000_ulBCU=dataBCU.field('eflux1000_ul95')
    
    lcTrain=createLCwithUL(tsTrain,ef1000Train,ef1000_ulTrain,indTrain)
    lcVali=createLCwithUL(tsVali,ef1000Vali,ef1000_ulVali,indVali)
    
    x_trainData = []
    y_train = []
    
    for source in indTrain:
        if source in indTrainBLL:
            x_trainData.append(lcTrain[source])
            y_train.append(0)
        elif source in indTrainFSRQ:
            x_trainData.append(lcTrain[source])
            y_train.append(1)

    x_valiData = []
    y_vali = []
    for source in indVali:
        if source in indValiBLL:
            x_valiData.append(lcVali[source])
            y_vali.append(0)
        elif source in indValiFSRQ:
            x_valiData.append(lcVali[source])
            y_vali.append(1)
    
    for eps in np.linspace(0.1, 4.0, num=2):
        for steps in np.arange(1,3,1):
            neural_network(eps,steps,x_trainData,y_train,x_valiData,y_vali)

    hdulTrain.close()
    hdulVali.close()
    #hdulTest.close()
    #hdulBCU.close()

#Execute the main function
if __name__ == "__main__":
    main()
