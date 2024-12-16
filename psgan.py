import tensorflow as tf
from tensorflow.keras import layers
from config import Config
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
        if self.config.nz_periodic == 0: # Skip if no periodic noise required.
            return Z
        
        nPeriodic = self.config.nz_periodic # Determine number of noise dimensions to add
        batch_size, _, zx, _ = Z.shape # Extract the spatial dimension

        periodic_noise = []
        for i in range(1, nPeriodic + 1): # Add periodic noise to input Z
            freq = (0.5 * i / nPeriodic) + 0.5 # Frequency of sinusoidla noise
            x_indices = tf.range(zx, dtype=tf.float32) * freq
            y_indices = tf.range(zx, dtype=tf.float32) * freq 
            x_indices = tf.sin(x_indices)[:, None]
            y_indices = tf.cos(y_indices)[None, :]
            periodic_noise.append(x_indices + y_indices)

        periodic_noise = tf.stack(periodic_noise, axis=-1)
        periodic_noise = tf.broadcast_to(periodic_noise, Z.shape)
        return tf.concat([Z, periodic_noise], axis=1)
    
class Generator(tf.keras.Model):
    '''
    Generator for PSGAN. Takes in noise, generates images.
    '''
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config 

        self.periodic_layer = PeriodicLayer(config) # Add the periodic noise layer like above
        self.transposed_conv_layers = [] # For upsampling
        self.batch_norm_layers = []
        for filters, kernel_size in zip(config.gen_fn[:-1], config.gen_ks):
            self.transposed_conv_layers.append( # Add convolutional layers + batchnorm
                layers.Conv2DTranspose(filters, kernel_size, strides=(2,2), padding='same', activation='relu')
            )
            self.batch_norm_layers.append(layers.BatchNormalization()) # add batchnorm layers

        self.final_layer = layers.Conv2DTranspose( # Final output layer from paper — tanh 
            config.gen_fn[-1], config.gen_ks[-1], strides=(2, 2), padding="same", activation="tanh"
        )

    def call(self, Z):
        x = self.periodic_layer(Z) # Periodic layer --> conv --> bn --> final
        for conv, bn in zip(self.transposed_conv_layers, self.batch_norm_layers):
            x = conv(x)
            x = bn(x) # Alternate convolution and batch norm 
        return self.final_layer(x)
    
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
        self.final_layer = layers.Dense(1, activation='sigmoid')

    def call(self, X):
        x = X # Similarly, forward pass
        for conv, bn in zip(self.conv_layers, self.batch_norm_layers):
            x = conv(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = bn(x)
        x = self.flatten(x)
        return self.final_layer(x)