from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.utils import plot_model

'''
This architecture is heavily inspired from the one used in the paper:
Variational End-to-End Navigation and Localization
    - Alexander Amini, and others.
'''


def DriveNet():
    minimap_input = Input(shape=(50, 50, 1))
    screen_input = Input(shape=(80, 200, 3))

    screen = Conv2D(filters=24, kernel_size=5, strides=2, activation='relu')(screen_input)
    screen = Conv2D(filters=36, kernel_size=5, strides=2, activation='relu')(screen)
    screen = Conv2D(filters=48, kernel_size=3, strides=2, activation='relu')(screen)
    screen = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(screen)
    screen = Flatten()(screen)

    minimap = Conv2D(filters=24, kernel_size=5, strides=2, activation='relu')(minimap_input)
    minimap = Conv2D(filters=36, kernel_size=5, strides=2, activation='relu')(minimap)
    minimap = Conv2D(filters=48, kernel_size=3, strides=2, activation='relu')(minimap)
    minimap = Flatten()(minimap)

    merged = Concatenate()([minimap, screen])

    x = Dense(1000, activation='relu')(merged)
    x = Dense(100, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)

    # minimap is the first input, followed by screen
    model = Model(inputs=[minimap_input, screen_input], outputs=output)

    return model


if __name__ == '__main__':
    model = DriveNet()

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    plot_model(model, to_file='DriveNet.png', show_shapes=True,
               show_layer_names=True, dpi=200)
