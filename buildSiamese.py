from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras import regularizers
def build_siamese(inputShape, embeddingDim = 48):
    inputs = Input(inputShape, name="")

    #define two sets of rules for our sister networks
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    l1_strength = 0.001  # Adjust this value based on the degree of regularization you need
    l2_strength = 0.001
    # Dense layers
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength))(x)
    #output
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)

    model = Model(inputs, outputs)
    return model