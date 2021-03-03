!pip install -q tensorflow-gpu==2.3.0-rc1
!pip install -q imageio
!pip install gast==0.3.3

import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import pathlib
import random
from IPython import display

print(tf.__version__)
print(tf.test.gpu_device_name())

AUTOTUNE = tf.data.experimental.AUTOTUNE

BUFFER_SIZE = 60000
BATCH_SIZE = 16

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

GRADIENT_PENALTY_WEIGHT = 0.5

IMAGE_SIZE = 128

seed = tf.random.normal([num_examples_to_generate, noise_dim])

class ResidualBlock(tf.keras.Model):
    def __init__(self, filter_size, activation):
        super(ResidualBlock, self).__init__()
        filter_size

        self.conv2a = tf.keras.layers.Conv2D(filter_size, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filter_size, (3, 3), padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filter_size*4, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

        self.activation = None
        if activation == "relu":
            self.activation = tf.keras.layers.ReLU()
        elif activation == "leaky_relu":
            self.activation = tf.keras.layers.LeakyReLU()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        x = self.activation(x, training=training)
        return tf.nn.leaky_relu(x)



def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense((IMAGE_SIZE//4)*(IMAGE_SIZE//4)*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((IMAGE_SIZE//4, IMAGE_SIZE//4, 256)))

    model.add(ResidualBlock(64, "relu"))
    model.add(ResidualBlock(64, "relu"))

    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, IMAGE_SIZE//2, IMAGE_SIZE//2, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding="same", use_bias=False, activation="tanh"))
    assert model.output_shape == (None, IMAGE_SIZE, IMAGE_SIZE, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(128, (1, 1), strides=(1, 1), padding="same",
                                     input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(ResidualBlock(32, "leaky_relu"))
    model.add(ResidualBlock(32, "leaky_relu"))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator = make_generator_model()
discriminator = make_discriminator_model()


#This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        alpha = tf.random.normal([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)   #Gradient penalty
        diff = images - generated_images
        interpolated = images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = discriminator(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output) + gp * GRADIENT_PENALTY_WEIGHT

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        #Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)

        #Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ("Time for epoch {} is {} sec".format(epoch + 1, time.time()-start))

    #Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           seed)


def generate_and_save_images(model, epoch, test_input):
    #Notice `training` is set to False.
    #This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    dpi = 15
    fig = plt.figure(figsize=((IMAGE_SIZE)/dpi,(IMAGE_SIZE)/dpi))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(tf.cast(predictions[i, :, :, :] * 127.5 + 127.5, tf.int32))
        plt.axis("off")

    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    plt.show()


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = (image - 127.5) / 127.5  # normalize to [-1,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)
  

data_root = pathlib.Path("/content/drive/MyDrive/CelebASmall")

all_image_paths = list(data_root.glob("*"))
all_image_paths = all_image_paths[:3000]
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
train_dataset = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

#Batch and shuffle the data
train_dataset = train_dataset.shuffle(buffer_size=len(all_image_paths))
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)


#This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


generator_optimizer = tf.keras.optimizers.Adam(0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)


checkpoint_dir = "/content/drive/MyDrive/Training_Checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open("image_at_epoch_{:04d}.png".format(epoch_no))

display_image(EPOCHS)

anim_file = "dcgan.gif"

with imageio.get_writer(anim_file, mode="I") as writer:
    filenames = glob.glob("image*.png")
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)


