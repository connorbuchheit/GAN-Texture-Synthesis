import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from config import Config
from psgan import Generator, Discriminator, NoiseGenerator
import homography
import os
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_and_preprocess_single_image, load_and_preprocess_images, visualize_dataset_images

config = Config() # Initialize configuration of NN
generator = Generator(config)
discriminator = Discriminator(config) # Initialize generator and discriminator models
wd_mult_gen = 0 # 0.0001 # weight decay multiplier
wd_mult_disc = 0 # 0.0001
noise_gen = NoiseGenerator(config)

gen_optimizer = Adam(learning_rate=config.lr, beta_1=config.b1) # Initialize both optimiizers
disc_optimizer = Adam(learning_rate=(config.lr * 0.1), beta_1=config.b1) # Keep separate due to two diff objective functions

bce_loss = BinaryCrossentropy(from_logits=True) # from_logits for more stablity w sigmoid

def discriminator_loss(real_output, fake_output, gradient_penalty, lambda_gp=10):
    """
    WGAN-GP discriminator loss with gradient penalty.
    """
    wasserstein_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
    total_loss = wasserstein_loss + lambda_gp * gradient_penalty
    return total_loss


def generator_loss(fake_output):
    """
    WGAN-GP generator loss.
    """
    return -tf.reduce_mean(fake_output)

def compute_gradient_penalty(discriminator, real_images, fake_images):
    """
    Compute the gradient penalty for WGAN-GP.
    """
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)  # Random interpolation factor
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        interpolated_output = discriminator(interpolated_images)
    
    gradients = tape.gradient(interpolated_output, [interpolated_images])[0]
    gradients_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((gradients_l2 - 1.0) ** 2)
    return gradient_penalty

def save_generated_images(images, epoch, samples_dir='samples_chequered'):
    images = (images + 1.0) * 127.5
    images = tf.clip_by_value(images, 0, 255).numpy().astype(np.uint8)
    os.makedirs(samples_dir, exist_ok=True)
    for i, img in enumerate(images[:5]):
        tf.keras.preprocessing.image.save_img(
            f"{samples_dir}/generated_epoch_{epoch + 1}_img_{i + 1}.png", img
        )

os.makedirs('models_chequered', exist_ok=True) # Create directory to write stuff to to reuse
os.makedirs('samples_chequered', exist_ok=True)

# fixed_noise = noise_gen.generate_noise(batch_size=25) # Generate noise
# fixed_noise = noise_gen.generate_noise(batch_size=1)

def log_gradients_to_file(step, gradients, file_path='gradient_logs.txt', label="Gradients"):
    with open(file_path, "a") as f:  # Open in append mode
        f.write(f"Step {step} - {label}:\n")
        for i, grad in enumerate(gradients):
            if grad is not None:
                mean_grad = tf.reduce_mean(grad).numpy()
                f.write(f"  Layer {i}: Mean Gradient = {mean_grad}\n")
        f.write("\n")

@tf.function
def train_step(real_images, train_gen_only=False):
    '''
    Perform one training step for PSGAN, which entails:
    1) Updating the discriminator.
    2) Updating the generator.
    '''
    # noise = noise_gen.generate_noise(batch_size=25)
    # noise = noise_gen.generate_noise(batch_size=1)
    # print(f"Noise Shape: {tf.shape(noise)}")

    # Generate noise
    # noise = noise_gen.generate_noise(batch_size = config.batch_size, shape = config.spatial_size)

    # Update discriminator
    if not train_gen_only:
        with tf.GradientTape() as dg:
            gen_images = generator(training=True)
            real_output = discriminator(real_images, training=True) # Check discernment
            fake_output = discriminator(gen_images, training=True)

            # real_loss = bce_loss(tf.ones_like(real_output), real_output)
            # fake_loss = bce_loss(tf.zeros_like(fake_output), fake_output) # Real: 1s, fake: 0s
            # discriminator_kernels = discriminator.trainable_variables  # Replace with the kernel weights of the discriminator
            # g_wd = wd_mult_gen * tf.add_n([tf.nn.l2_loss(kernel) for kernel in discriminator_kernels])
            # disc_loss = g_wd + real_loss + fake_loss

            # gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_images, gen_images)

            # Discriminator loss
            loss_d = discriminator_loss(real_output, fake_output, gradient_penalty)

            # homography layer doesn't do as well as omitting it... for now
            
            # inverse_homography_layer = discriminator.homography_layer
            # estimated_homography = inverse_homography_layer(real_images)
            # warped_fake_images = tf.map_fn(lambda x: homography.apply_homography(x, estimated_homography), gen_images)
            # reconstruction_loss = tf.reduce_mean(tf.square(real_images - warped_fake_images))
            total_discriminator_loss = loss_d #+ reconstruction_loss

        disc_grads = dg.gradient(total_discriminator_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        # log_gradients_to_file(step_no, disc_grads, label="Discriminator Gradients")
    else:
        total_discriminator_loss = None

    # Update generator
    with tf.GradientTape() as gg:
        gen_images = generator(training = True)
        gen_output = discriminator(gen_images, training = True)
        loss_g = generator_loss(gen_output)

    gen_grads = gg.gradient(loss_g, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    # log_gradients_to_file(step_no, gen_grads, label="Generator Gradients")
    return loss_g, total_discriminator_loss, gen_images # Return losses AND geneated images 

def train(dataset, epochs, log_file="gradient_logs.txt"):
    '''
    Train PSGAN model for given # of epochs
    '''
    gen_losses = []
    disc_losses = []
    step_counter = 0 
    with open(log_file, "w") as f:
        f.write("Gradient Logs:\n\n")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_gen_only = (step_counter % 5 != 0)
        for step, real_images in enumerate(dataset):
            gen_loss, disc_loss, gen_images = train_step(real_images, train_gen_only) # Go through dataset
            gen_losses.append(gen_loss); disc_losses.append(disc_loss)
            if step_counter % 100 == 0:
                print(f"Step {step}: Generator Loss: {gen_loss:.4f}, Disciminator loss: {disc_loss:.4f}")
        # save_generated_images(gen_images, epoch)
        step_counter += 1
        
        if (epoch + 1) % 25 == 0: # Save model weights per 100th iteration
            generator.save_weights(f"models_chequered/generator_epoch_{epoch + 1}.weights.h5")
            discriminator.save_weights(f"models_chequered/discriminator_epoch_{epoch + 1}.weights.h5")
            save_generated_images(gen_images, epoch)

            with open('gen_losses.txt', 'w') as gen_file:
                gen_file.writelines(f"{loss}\n" for loss in gen_losses)
        
            # Save discriminator losses
            with open('disc_losses.txt', 'w') as disc_file:
                disc_file.writelines(f"{loss}\n" for loss in disc_losses)
            
            print(f"Saved losses to files at epoch {epoch}")


    return gen_losses, disc_losses