import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def make_pairs(images, labels):
    pairImages = []
    pairLabels = []
    posCount, negCount = 0, 0
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(numClasses)]

    for class_idx in idx:
        for i in range(len(class_idx)):
            for j in range(i + 1, len(class_idx)):
                pairImages.append([images[class_idx[i]], images[class_idx[j]]])
                pairLabels.append(1)
                posCount += 1

                available_indices = np.arange(len(images))
                negative_indices = np.setdiff1d(available_indices, class_idx)  # Exclude positive class indices
                if len(negative_indices) > 0:
                    random_class_idx = np.random.choice(negative_indices)
                    pairImages.append([images[class_idx[i]], images[random_class_idx]])
                    pairLabels.append(0)
                    negCount += 1
    print( "Positive: " + str(posCount) + " Negative: " + str(negCount))
    return (np.array(pairImages),np.array(pairLabels))

def load_data(directory):
    images = []
    labels = []
    for path in os.listdir(directory):
        if path.endswith('.jpg'):
            image_path = os.path.join(directory, path)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
            img = cv2.resize(img, (28, 28))  # Resize THIS MIGHT BE A POINT OF FAILURE TO CHANGE
            images.append(img)

            parts = path.split('-')
            if len(parts) == 3:
                label = int(''.join(parts[:-1]))
            elif len(parts) == 2:
                num = parts[-1].split('.')
                label = int(''.join(parts[:-1]) + num[0])

            labels.append(int(label))
    return np.array(images), np.array(labels)

def euclidean_distance(vectors):
    (vecA, vecB) = vectors
    sumSquared = K.sum(K.square(vecA -vecB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def training_plot(H, plotPath):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)
