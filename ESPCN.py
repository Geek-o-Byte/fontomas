import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Lambda
from tensorflow.keras.models import Model

def subpixel_conv2d(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def ESPNet(input_shape, scale, num_filters):
    inputs = Input(shape=input_shape)

    # First convolutional layer
    x = Conv2D(num_filters, (5, 5), padding='same')(inputs)
    x = Activation('relu')(x)

    # Second convolutional layer
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    # Sub-pixel convolution layer
    x = Conv2D(num_filters * (scale ** 2), (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Lambda(subpixel_conv2d(scale))(x)

    # Output layer
    outputs = Conv2D(1, (1, 1), padding='same')(x)

    model = Model(inputs, outputs)
    return model

# Example usage
input_shape = (None, None, 1)  # Single channel input (e.g., grayscale image)
scale = 2  # Upscaling factor
num_filters = 64  # Number of filters in convolutional layers

model = ESPNet(input_shape, scale, num_filters)
model.compile(optimizer='adam', loss='mean_squared_error')
