import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model

def unet_encoder(input_shape=(32, 32, 3), num_filters=64):
    inputs = Input(input_shape)

    # Первый уровень
    c1 = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # 16x16

    # Второй уровень
    c2 = Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # 8x8

    # Третий уровень
    c3 = Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)  # 4x4

    # Создание модели
    model = Model(inputs, p3)
    return model

import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.activations import swish

def ClassConditioning(res, num_channels=1):
    def block(x):
        x = Dense(res * res * num_channels)(x)
        x = swish(x)  # Using the swish activation function (equivalent to SiLU)
        x = Reshape((res, res, num_channels))(x)
        return x
    return block

# Example usage:
input_shape = (64,)  # Assuming the input shape is (128,)
inputs = Input(shape=input_shape)
outputs = ClassConditioning(res=4, num_channels=256)(inputs)
model_ = Model(inputs, outputs)

# Compile the model
model_.compile()


model = unet_encoder()
model.compile(optimizer='adam', loss='mse')
model.summary()

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plot_model(model_, to_file='model_plot_1.png', show_shapes=True, show_layer_names=True)
