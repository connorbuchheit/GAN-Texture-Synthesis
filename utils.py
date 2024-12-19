import tensorflow as tf
import matplotlib.pyplot as plt

def load_and_preprocess_images(image_dir, target_size=(256, 256), batch_size=1):
    """
    Load images from a directory, crop to a fixed size, normalize to [-1, 1], and batch them.
    Inputs:
        image_dir (str): Path to the directory containing images.
        crop_size (tuple): Crop size for the images (height, width).
        batch_size (int): Batch size for loading.
        
    Outputs:
        Dataset: A TensorFlow dataset of preprocessed images.
    """
    # Load images WITHOUT resizing
    dataset = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        label_mode=None,  
        image_size=target_size, 
        batch_size=batch_size,
        shuffle=True
    )

    # Crop and normalize images
    dataset = dataset.map(lambda x: (x / 127.5) - 1.0, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetching improves performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def visualize_dataset_images(dataset, num_images=1):
    """
    Visualize a batch of images from the dataset.
    Inputs:
        dataset: TensorFlow dataset of images.
        num_images: Number of images to display.
    """
    # Extract one batch of images
    for images in dataset.take(1):
        images = (images + 1.0) / 2.0  # Rescale from [-1, 1] to [0, 1]
        images = tf.clip_by_value(images, 0.0, 1.0)  # Ensure no values outside [0, 1]
        
        plt.figure(figsize=(15, 5))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i].numpy())
            plt.axis("off")
            plt.title(f"Image {i+1}")
        plt.tight_layout()
        plt.show()
        break

def load_and_preprocess_single_image(image_path, crop_size=(128, 128)):
    """
    Load a single image, crop it to the center, normalize to [-1, 1], and wrap in a dataset.
    Inputs:
        image_path (str): Path to the image file.
        crop_size (tuple): Desired crop size (height, width).
    Outputs:
        A TensorFlow dataset containing the preprocessed image.
    """
    # Load the image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)  # Decode as RGB
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to [0, 1]
    
    # Get image dimensions
    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]
    
    # Calculate cropping box for center crop
    crop_height, crop_width = crop_size
    offset_height = (original_height - crop_height) // 2
    offset_width = (original_width - crop_width) // 2

    cropped_image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width
    )
    
    # Normalize to [-1, 1]
    cropped_image = (cropped_image * 2.0) - 1.0
    
    # Wrap in a dataset
    dataset = tf.data.Dataset.from_tensors(cropped_image).batch(1).prefetch(tf.data.AUTOTUNE)
    return dataset
