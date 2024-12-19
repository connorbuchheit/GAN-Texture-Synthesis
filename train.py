import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from config import Config
from psgan import Generator, Discriminator, NoiseGenerator
import os
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_and_preprocess_single_image, load_and_preprocess_images, visualize_dataset_images

config = Config() # Initialize configuration of NN
generator = Generator(config)
discriminator = Discriminator(config) # Initialize generator and discriminator models

gen_optimizer = Adam(learning_rate=config.lr, beta_1=config.b1) # Initialize both optimiizers
disc_optimizer = Adam(learning_rate=config.lr, beta_1=config.b1) # Keep separate due to two diff objective functions

bce_loss = BinaryCrossentropy(from_logits=True) # from_logits for more stablity w sigmoid


def save_generated_images(images, epoch, samples_dir='samples_honeycomb'):
    images = (images + 1.0) * 127.5
    images = tf.clip_by_value(images, 0, 255).numpy().astype(np.uint8)
    os.makedirs(samples_dir, exist_ok=True)
    for i, img in enumerate(images[:5]):
        tf.keras.preprocessing.image.save_img(
            f"{samples_dir}/generated_epoch_{epoch + 1}_img_{i + 1}.png", img
        )

os.makedirs('models_honeycomb', exist_ok=True) # Create directory to write stuff to to reuse
os.makedirs('samples_honeycomb', exist_ok=True)

noise_gen = NoiseGenerator(config)

# fixed_noise = noise_gen.generate_noise(batch_size=25) # Generate noise
fixed_noise = noise_gen.generate_noise(batch_size=1)

@tf.function
def train_step(real_images):
    '''
    Perform one training step for PSGAN, which entails:
    1) Updating the discriminator.
    2) Updating the generator.
    '''
    # noise = noise_gen.generate_noise(batch_size=25)
    noise = noise_gen.generate_noise(batch_size=1)
    print(f"Noise Shape: {tf.shape(noise)}")

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
        # save_generated_images(gen_images, epoch)
        
        if (epoch + 1) % 25 == 0: # Save model weights per 100th iteration
            generator.save_weights(f"models_honeycomb/generator_epoch_{epoch + 1}.weights.h5")
            discriminator.save_weights(f"models_honeycomb/discriminator_epoch_{epoch + 1}.weights.h5")
            save_generated_images(gen_images, epoch)

if __name__ == "__main__":
    # Step 1) Need a method to preprocess dataset
    # AKA have dataset of images w predefined dimensions, 
    # Normalized between -1 and 1.
    # print(f"Expected dimension: {config.npx}")
    images_dir = Path(__file__).resolve().parent / "dtd_folder" / "dtd" / "images" / "z_training" 
    # image_path = Path(__file__).resolve().parent / "dtd_folder" / "dtd" / "images" / "z_training" / "honeycombed_0003.jpg"
    print("Loading images!")
    dataset = load_and_preprocess_images(images_dir, target_size=(160, 160))
    visualize_dataset_images(dataset)

    print("Starting PSGAN training...wish me luck")
    train(dataset, epochs=config.epoch_count)
    print("Training complete! Check 'samples/' for generated images and 'models/' for saved models.")
