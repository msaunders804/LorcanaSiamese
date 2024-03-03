import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, losses, optimizers
from PIL import Image
import matplotlib.pyplot as plt

# Define the Siamese Network architecture
def create_siamese_network(input_shape):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    for layer in base_model.layers:
        layer.trainable = False

    flatten = layers.Flatten()(base_model.output)
    dense = layers.Dense(128, activation="relu")(flatten)
    output = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense)

    return Model(inputs=base_model.input, outputs=output)

# Define the triplet loss function
class TripletLoss(losses.Loss):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)

# Load data
def load_data(directory, target_size=(224, 312)):
    images = []
    labels = []
    for path in os.listdir(directory):
        if path.endswith('.jpg'):
            image_path = os.path.join(directory, path)
            img = Image.open(image_path)
            img = img.convert('RGB')
            resized_image = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(resized_image)
            if len(img_array.shape) ==3:
                images.append(np.array(resized_image))
                label = int(path.split('-')[0])  # Assuming label is the first part before '-'
                labels.append(label)
            else:
                print(img_array.shape)
    return np.array(images, dtype=float), np.array(labels)

# Generate triplets
def make_triplets(images, labels):
    triplets = []
    num_classes = len(np.unique(labels))
    indices = [np.where(labels == i)[0] for i in range(num_classes)]

    for class_idx in indices:
        for i, anchor_idx in enumerate(class_idx):
            for j in range(i+1, len(class_idx)):
                positive_idx = class_idx[i]
                negative_idx = np.random.choice(np.setdiff1d(class_idx, [anchor_idx, positive_idx]))
                triplets.append([images[anchor_idx], images[positive_idx], images[negative_idx]])
    return np.array(triplets)

# Siamese network model
def build_siamese(input_shape):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    return Model(inputs=input_layer, outputs=x)

# Training plot
def training_plot(history, plot_path):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)

# Load data and create triplets
train_path = input("Enter the path of the training images: ")
test_path = input("Enter the path of the test images: ")
trainX, trainY = load_data(train_path)
testX, testY = load_data(test_path)
trainX /= 255.0
testX = np.float32(testX)
testX /= 255.0
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

tripletsTrain = make_triplets(trainX, trainY)
#tripletsTrain_flat = tripletsTrain.reshape(-1, tripletsTrain.shape[-3], tripletsTrain.shape[-2], tripletsTrain.shape[-1])
tripletsTrain_tensor = tf.convert_to_tensor(tripletsTrain, dtype=tf.float32)
tripletsTest = make_triplets(testX, testY)
#tripletsTest_flat = tripletsTest.reshape(-1, tripletsTest.shape[-3], tripletsTest.shape[-2], tripletsTest.shape[-1])
tripletsTest_tensor = tf.convert_to_tensor(tripletsTest, dtype=tf.float32)

# Create and compile the Siamese network
input_shape = (312,224, 3)  # Adjust based on your image size
siamese_net = create_siamese_network(input_shape)
siamese_net.compile(optimizer=optimizers.Adam(), loss=TripletLoss())

# Train the Siamese network
history = siamese_net.fit([tripletsTrain_tensor[:, :, :, 0],
                           tripletsTrain_tensor[:, :, :, 1],
                           tripletsTrain_tensor[:, :, :, 2]],
                          np.zeros(len(tripletsTrain_tensor)),
                          validation_data=([tripletsTest_tensor[:, :, :, 0],
                                            tripletsTest_tensor[:, :, :, 1],
                                            tripletsTest_tensor[:, :, :, 2]],
                                           np.zeros(len(tripletsTest_tensor))),
                          batch_size=32, epochs=10)

# Save the model
siamese_net.save("siamese_model.h5")

# Plot training history
training_plot(history, "training_plot.png")

# Inference: Given a new trading card, compute its embedding and compare with others
# Enjoy identifying similar trading cards!
