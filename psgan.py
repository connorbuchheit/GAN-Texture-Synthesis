import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from config import Config
# Inspired from "Learning Texture Manifolds with the Periodic Spatial GAN” by Bergmann et al., 2017,
# updated in TensorFlow rather than LASAGNE because modern technology rules

class NoiseGenerator:
    """
    Layer to generate periodic noise maps and concatenate them with input tensors.
    """
    def __init__(self, config):
        self.dim_z_periodic = config.dim_z_periodic
        self.dim_z_local = config.dim_z_local
        self.spatial_size = config.spatial_size # INIITALIZED IN CONFIG AS EXAMPLE: CHANGE IF NEEDED
        self.K = tf.Variable(
            initial_value=tf.random.normal([2, self.dim_z_periodic], stddev=0.1),
            trainable=True,
            name="wave_numbers"
        )
            
    def generate_local_noise(self, batch_size, shape=None):
        if shape is None:
            shape = self.spatial_size
        height, width = shape
        local_noise = tf.random.uniform(
            shape=(batch_size, height, width, self.dim_z_local),
            minval=-1.0, maxval=1.0
        )
        return local_noise
    
    def generate_periodic_noise(self, batch_size, shape=None):
        if shape is None:
            shape = self.spatial_size 
        height, width = shape
        x = tf.range(width, dtype=tf.float32)
        y = tf.range(height, dtype=tf.float32)
        x, y = tf.meshgrid(x, y)

        # Should be of shape [height, width, 2]
        coordinates = tf.stack([x, y], axis=-1)

        # Random phase offsets from 0 to 2π
        phi = tf.random.uniform(
            shape=(self.dim_z_periodic,), minval=0.0, maxval=2 * np.pi
        )  # Should be of shape [dim_z_periodic]

        # Compute the dot product between wave numbers K and coordinates
        # Shape of self.K: [2, dim_z_periodic]
        # Resulting shape: [height, width, dim_z_periodic] 
        wave_patterns = tf.tensordot(coordinates, self.K, axes=1)

        # Add phase offsets
        wave_patterns = wave_patterns + phi  # Broadcast phi across spatial dimensions

        # Apply sinusoid
        periodic_maps = tf.sin(wave_patterns)  # Shape: [height, width, dim_z_periodic]

        # Expand batch dimension and tile
        periodic_maps = tf.tile(
            periodic_maps[None, :, :, :], [batch_size, 1, 1, 1]
        )  # Shape: [batch_size, height, width, dim_z_periodic]

        return periodic_maps

    def generate_noise(self, batch_size, shape=None):
        if shape is None:
            shape = self.spatial_size
        local_noise = self.generate_local_noise(batch_size, shape)
        periodic_noise = self.generate_periodic_noise(batch_size, shape)
        return tf.concat([local_noise, periodic_noise], axis=-1)


class Generator(tf.keras.Model):
    '''
    Generator for PSGAN. Takes in noise, generates images.
    '''
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config 
        self.noise_gen = NoiseGenerator(config)
        
        self.conv_layers = [ # Zickler conv layers
            # layers.Conv2DTranspose(512, (5, 5), strides=(2,2), padding='valid', activation='relu'),
            layers.Conv2DTranspose(256, (5, 5), strides=(2,2), padding='valid', activation='relu', kernel_initializer='glorot_uniform'),
            layers.Conv2DTranspose(128, (5, 5), strides=(2,2), padding='valid', activation='relu', kernel_initializer='glorot_uniform'),
            layers.Conv2DTranspose(64, (5, 5), strides=(2,2), padding='valid', activation='relu', kernel_initializer='glorot_uniform'),
            # layers.Conv2DTranspose(32, (5, 5), strides=(2,2), padding='valid', activation='relu')
        ]

        self.batch_norm_layers = [layers.BatchNormalization() for _ in range(len(self.conv_layers))]
        self.final_layer = layers.Conv2DTranspose(
            3, (5,5), strides=(2,2), padding='same', activation='tanh'
        )


    def call(self, training=False):
        x = self.noise_gen.generate_noise(self.config.batch_size, shape=self.config.spatial_size)
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norm_layers)):
            # print(x.shape)
            x = conv(x)
            x = bn(x, training=training)
            x = x[:, 3:-3, 3:-3, :] # crop 3 pixels from each side bufferwise
            current_shape = x.shape[1:3]  # Extract height and width
            
            # Add periodic noise after the first layer (and train on periodic noise)
            if i == 0:
                periodic_noise = self.noise_gen.generate_periodic_noise(x.shape[0], shape=current_shape)
                x = tf.concat([x, periodic_noise], axis=-1)
            
            # Add local noise after the second and third layers
            if i in [1, 2]:
                local_noise = self.noise_gen.generate_local_noise(x.shape[0], shape=current_shape)
                x = tf.concat([x, local_noise], axis=-1)

        # print(f"After loop: {x.shape}")
        x = self.final_layer(x)
        # print(f"Final:{x.shape}")
        return x[:, 3:-3, 3:-3, :]

class Discriminator(tf.keras.Model):
    '''
    Discriminator for PSGAN. Classifies input as real or fake.
    '''
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.conv_layers = [ # Zickler conv layers
            layers.Conv2D(64, (5, 5), strides=(2,2), padding='same', activation=None),
            layers.Conv2D(128, (5, 5), strides=(2,2), padding='same', activation=None),
            layers.Conv2D(256, (5, 5), strides=(2,2), padding='same', activation=None),
        ]

        self.batch_norm_layers = [None] + [layers.BatchNormalization() for _ in range(len(self.conv_layers) - 1)]
        # self.batch_norm_layers = [layers.BatchNormalization() for _ in range(len(self.conv_layers))]

        self.flatten = layers.Flatten() # Final classification layer for probablity
        self.final_layer = layers.Dense(1) # Removed sigmoid here in favor of applying it in BinaryCrossentropy in train.py (logits=True)

    def call(self, X):
        x = X # Similarly, forward pass
        for conv, bn in zip(self.conv_layers, self.batch_norm_layers):
            x = conv(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            if bn:
                x = bn(x)
            # print(f"Shape after conv and batch norm: {x.shape}")
        x = self.flatten(x)
        # print(f"Shape after flattening: {x.shape}")  # Debugging
        return self.final_layer(x)
    
config = Config()

# Initialize Generator
generator = Generator(config)

# Pass noise through the generator
output = generator(training=False)
print("Generated output shape:", output.shape)