import numpy as np

import buildSiamese
import config
import buildPairs
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda

def trainSiamese():
    #Loading data and creating pairs
    train_path = input("Enter the path of the training images: ")
    test_path = input("Enter the path of the test images: ")
    trainX, trainY = buildPairs.load_data(train_path)
    testX, testY = buildPairs.load_data(test_path)
    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)

    pairTrain, labelTrain = buildPairs.make_pairs(trainX, trainY)
    pairTest, labelTest = buildPairs.make_pairs(trainX, testY)

    #Generate sister networks
    imgA = Input(shape=config.IMG_SHAPE)
    imgB = Input(shape=config.IMG_SHAPE)
    featureExtract = buildSiamese.build_siamese(config.IMG_SHAPE)
    vecA = featureExtract(imgA)
    vecB = featureExtract(imgB)

    distance = Lambda(buildPairs.euclidean_distance)([vecA,vecB])
    outputs = Dense(1, activation='sigmoid')(distance)
    model = Model(inputs=[imgA, imgB], outputs=outputs)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit([pairTrain[:,0], pairTrain[:,1]], labelTrain[:],
                        validation_data=([pairTest[:,0], pairTest[:,1]], labelTest[:]),
                        batch_size= config.BATCH_SIZE, epochs=config.EPOCHS)
    model.save(config.MODEL_PATH)
    buildPairs.training_plot(history, config.PLOT_PATH)
