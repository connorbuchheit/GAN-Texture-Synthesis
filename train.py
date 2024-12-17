import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from configs import Config
from psgan import Generator, Discriminator
import os

config = Config() # Initialize configuration of NN
generator = Generator(config)
discriminator = Discriminator(config) # Initialize generator and discriminator models

gen_optimizer = Adam(learning_rate=config.lr, beta_1=config.b1) # Initialize both optimiizers
disc_optimizer = Adam(learning_rate=config.lr, beta_1=config.b1) # Keep separate due to two diff objective functions

bce_loss = BinaryCrossentropy(from_logits=True) # from_logits for more stablity w sigmoid

def generate_noise(batch_size, nz, zx): # Function for input noise
    return tf.random.normal([batch_size, nz, zx, zx])

fixed_noise = generate_noise(1, config.nz, config.zx_sample) # Generate noise

os.makedirs('samples', exist_ok=True) # Create directory to write stuff to to reuse
os.makedirs('models', exist_ok=True)

@tf.function
def train_step(real_images):
    '''
    Perform one training step for PSGAN, which entails:
    1) Updating the discriminator.
    2) Updating the generator.
    '''
    noise = generate_noise(config.batch_size, config.nz, config.zx) # Make noise

    # Update discriminator
    with tf.GradientTape() as gg:
        gen_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True) # Check discernment
        fake_output = discriminator(gen_images, training=True)

        real_loss = bce_loss(tf.ones_like(real_output), real_output)
        fake_loss = bce_loss(tf.zeros_like(fake_output), fake_output) # Real: 1s, fake: 0s
        disc_loss = real_loss + fake_loss 

    disc_grads = gg.gradient(disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # Update generator
    with tf.GradientTape() as dg:
        gen_images = generator(noise, training=True)
        fake_output = discriminator(gen_images, training=True)
        gen_loss = bce_loss(tf.ones_like(fake_output), fake_output) # Generator wins if discriminator says its real

    gen_grads = dg.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    return gen_loss, disc_loss 

def train(dataset, epochs):
    '''
    Train PSGAN model for given # of epochs
    '''
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        gen_losses = []
        disc_losses = []
        for step, real_images in enumerate(dataset):
            gen_loss, disc_loss = train_step(real_images) # Go through dataset
            gen_losses.append(gen_loss); disc_losses.append(disc_loss)
            if step % 100 == 0:
                print(f"Step {step}: Generator Loss: {gen_loss:.4f}, Disciminator loss: {disc_loss:.4f}")
        gen_images = generator(fixed_noise, training=False)
        
        if (epoch + 1) % 10 == 0: # Save model weights per 10th iteration
            generator.save_weights(f"models/generator_epoch_{epoch + 1}")
            discriminator.save_weights(f"models/discriminator_epoch_{epoch + 1}")

