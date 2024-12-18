import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
# Inspired from "Learning Texture Manifolds with the Periodic Spatial GAN” by Bergmann et al., 2017,
# updated in TensorFlow rather than LASAGNE because modern technology rules

class PeriodicLayer(tf.keras.layers.Layer):
    '''
    The custom layer defined in the paper to add periodic noise to input tensor
    Meant to help GAN generator learn textures with periodic patterns.
    '''
    def __init__(self, config):
        '''
        Config params defined in config.py — used same ones in paper.
        '''
        super().__init__()
        self.config = config 

    def call(self, Z): 
        '''
        Method to add periodic noise to Z.
        Input: Z—input tensor of shape [batch_size, channels, height, width]
        Output: Tensor with additional periodic noise concatenated
        '''
        if self.config.nz_periodic == 0:
            return Z

        nPeriodic = self.config.nz_periodic
        batch_size, channels, zx, _ = Z.shape

        periodic_noise = []
        for i in range(1, nPeriodic + 1):
            freq = (0.5 * i / nPeriodic) + 0.5
            x_indices = tf.range(zx, dtype=tf.float32) * freq
            y_indices = tf.range(zx, dtype=tf.float32) * freq

            sin_wave = tf.sin(x_indices)[:, None] + tf.sin(y_indices)[None, :]
            cos_wave = tf.cos(x_indices)[:, None] + tf.cos(y_indices)[None, :]
            periodic_noise.extend([sin_wave, cos_wave])

        periodic_noise = tf.stack(periodic_noise, axis=0)  # Shape: (2 * nPeriodic, zx, zx)
        periodic_noise = periodic_noise[None, :, :, :]  # Add batch dimension
        periodic_noise = tf.tile(periodic_noise, [batch_size, 1, 1, 1])  # Repeat for batch size
        periodic_noise += tf.random.uniform(periodic_noise.shape) * 2 * np.pi  # Add random phase

        return tf.concat([Z, tf.sin(periodic_noise)], axis=1)  # Concatenate with Z

class Generator(tf.keras.Model):
    '''
    Generator for PSGAN. Takes in noise, generates images.
    '''
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config 

        self.periodic_layer = PeriodicLayer(config) # Add the periodic noise layer like above
        
        self.conv_layers = [ # Zickler conv layers
            layers.Conv2DTranspose(512, (5, 5), strides=(2,2), padding='valid', activation='relu'),
            layers.Conv2DTranspose(256, (5, 5), strides=(2,2), padding='valid', activation='relu'),
            layers.Conv2DTranspose(128, (5, 5), strides=(2,2), padding='valid', activation='relu'),
            layers.Conv2DTranspose(64, (5, 5), strides=(2,2), padding='valid', activation='relu'),
        ]

        self.batch_norm_layers = [layers.BatchNormalization() for _ in range(len(self.conv_layers))]
        self.final_layer = layers.Conv2DTranspose(
            3, (5,5), strides=(2,2), padding='valid', activation='tanh'
        )

    def crop_size(self, tensor, target_height, target_width): # crop like Zickler
        input_shape = tf.shape(tensor)
        crop_height = (input_shape[1] - target_height) // 2 
        crop_width = (input_shape[2] - target_width) // 2
        return tf.image.crop_to_bounding_box(tensor, crop_height, crop_width, target_height, target_width)


    def call(self, Z):
        x = self.periodic_layer(Z) # Periodic layer --> conv --> bn --> final
        # for conv, bn in zip(self.transposed_conv_layers, self.batch_norm_layers):
        #     x = conv(x)
        #     x = bn(x) # Alternate convolution and batch norm 
        # x = self.final_layer(x)
        current_size = self.config.initial_size + 3
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norm_layers)):
            x = conv(x)
            current_size = 2 * current_size - 3 
            print(f"Layer {i}: Expected size = {current_size}, Actual tensor size = {x.shape[1]}x{x.shape[2]}")
            x = self.crop_size(x, current_size, current_size)
            x = bn(x)
        x = self.final_layer(x)
        # final_size = 2 ** len(self.conv_layers) * self.config.initial_size
        final_size = 128 # rather arbitrarily chosen.
        x = self.crop_size(x, final_size, final_size)
        print(f"Generator Output Shape: {x.shape}")
        return x

class Discriminator(tf.keras.Model):
    '''
    Discriminator for PSGAN. Classifies input as real or fake.
    '''
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.conv_layers = []
        self.batch_norm_layers = []

        for filters, kernel_size in zip(config.dis_fn, config.dis_ks):
            self.conv_layers.append( # Similar architecture as Generator 
                layers.Conv2D(filters, kernel_size, strides=(2, 2), padding="same", activation=None)
            )
            self.batch_norm_layers.append(layers.BatchNormalization())

        self.flatten = layers.Flatten() # Final classification layer for probablity
        self.final_layer = layers.Dense(1) # Removed sigmoid here in favor of applying it in BinaryCrossentropy in train.py (logits=True)

    def call(self, X):
        x = X # Similarly, forward pass
        for conv, bn in zip(self.conv_layers, self.batch_norm_layers):
            x = conv(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = bn(x)
            print(f"Shape after conv and batch norm: {x.shape}")
        x = self.flatten(x)
        print(f"Shape after flattening: {x.shape}")  # Debugging
        return self.final_layer(x)