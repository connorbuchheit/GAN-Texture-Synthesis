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
wd_mult_gen = 0 # 0.0001 # weight decay multiplier
wd_mult_disc = 0 # 0.0001
noise_gen = NoiseGenerator(config)

gen_optimizer = Adam(learning_rate=config.lr, beta_1=config.b1) # Initialize both optimiizers
disc_optimizer = Adam(learning_rate=(config.lr * 0.1), beta_1=config.b1) # Keep separate due to two diff objective functions

bce_loss = BinaryCrossentropy(from_logits=True) # from_logits for more stablity w sigmoid


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

    # Update discriminator
    gen_images = generator(training=True)
    if not train_gen_only:
        with tf.GradientTape() as dg:
            real_output = discriminator(real_images, training=True) # Check discernment
            fake_output = discriminator(gen_images, training=True)

            real_loss = bce_loss(tf.ones_like(real_output), real_output)
            fake_loss = bce_loss(tf.zeros_like(fake_output), fake_output) # Real: 1s, fake: 0s
            discriminator_kernels = discriminator.trainable_variables  # Replace with the kernel weights of the discriminator
            g_wd = wd_mult_gen * tf.add_n([tf.nn.l2_loss(kernel) for kernel in discriminator_kernels])
            disc_loss = g_wd + real_loss + fake_loss

        disc_grads = dg.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        # log_gradients_to_file(step_no, disc_grads, label="Discriminator Gradients")
    else:
        disc_loss = None

    # Update generator
    with tf.GradientTape() as gg:
        fake_output = discriminator(gen_images, training=True)
        generator_kernels = generator.trainable_variables  # Replace with the kernel weights of the generator
        d_wd = wd_mult_disc * tf.add_n([tf.nn.l2_loss(kernel) for kernel in generator_kernels])
        gen_loss = d_wd + bce_loss(tf.ones_like(fake_output), fake_output)

    gen_grads = gg.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    # log_gradients_to_file(step_no, gen_grads, label="Generator Gradients")
    return gen_loss, disc_loss, gen_images # Return losses AND geneated images 

def train(dataset, epochs, log_file="gradient_logs.txt"):
    '''
    Train PSGAN model for given # of epochs
    '''
    step_counter = 0 
    with open(log_file, "w") as f:
        f.write("Gradient Logs:\n\n")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_gen_only = (step_counter % 5 != 0)
        gen_losses = []
        disc_losses = []
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

# if __name__ == "__main__":
#     # Step 1) Need a method to preprocess dataset
#     # AKA have dataset of images w predefined dimensions, 
#     # Normalized between -1 and 1.
#     # print(f"Expected dimension: {config.npx}")
#     images_dir = Path(__file__).resolve().parent / "dtd_folder" / "dtd" / "images" / "z_training" 
#     # image_path = Path(__file__).resolve().parent / "dtd_folder" / "dtd" / "images" / "z_training" / "honeycombed_0003.jpg"
#     print("Loading images!")
#     # dataset = load_and_preprocess_single_image(str(image_path), crop_size=(96, 96))
#     dataset = load_and_preprocess_images(images_dir, target_size=(160, 160))
#     visualize_dataset_images(dataset)

#     print("Starting PSGAN training...wish me luck")
#     train(dataset, epochs=config.epoch_count)
#     print("Training complete! Check 'samples/' for generated images and 'models/' for saved models.")
