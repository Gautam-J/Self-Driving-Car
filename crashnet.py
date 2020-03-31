from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import numpy as np

'''
AutoEncoder for detecting crashes and other anomalies.
'''


def CrashNet():
    road_map = Input(shape=(80, 200, 3))

    x = Conv2D(24, kernel_size=5, strides=2, padding='same', activation='relu')(road_map)
    x = BatchNormalization()(x)
    x = Conv2D(36, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    volumeSize = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(100)(x)

    encoder = Model(inputs=road_map, outputs=latent, name='encoder')

    latentInputs = Input(shape=(100,))

    x = Dense(np.prod(volumeSize[1:]))(latentInputs)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    x = Conv2DTranspose(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(48, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(36, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(24, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(latentInputs, outputs, name='decoder')

    autoencoder = Model(road_map, decoder(encoder(road_map)), name='autoencoder')

    return encoder, decoder, autoencoder


if __name__ == '__main__':
    encoder, decoder, autoencoder = CrashNet()
    encoder.summary(line_length=100)
    decoder.summary(line_length=100)
    autoencoder.summary(line_length=100)

    plot_model(encoder, to_file='CrashNet_encoder.png', show_shapes=True,
               show_layer_names=True, dpi=200)
    plot_model(decoder, to_file='CrashNet_decoder.png', show_shapes=True,
               show_layer_names=True, dpi=200)
    plot_model(autoencoder, to_file='CrashNet_autoencoder.png', show_shapes=True,
               show_layer_names=True, dpi=200, expand_nested=True)
