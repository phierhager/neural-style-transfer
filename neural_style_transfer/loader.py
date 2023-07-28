import tensorflow as tf
from PIL import Image
import numpy as np


def load_vgg(img_size):
    vgg = tf.keras.applications.VGG19(include_top=False,
                                      input_shape=(img_size, img_size, 3),
                                      weights='pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

    vgg.trainable = False
    return vgg


def get_content_img(img_size):
    content_image = np.array(Image.open(
        "images/louvre_small.jpg").resize((img_size, img_size)))
    content_image = tf.constant(np.reshape(
        content_image, ((1,) + content_image.shape)))
    return (content_image)


def get_style_img(img_size):
    style_image = np.array(Image.open(
        "images/monet.jpg").resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(
        style_image, ((1,) + style_image.shape)))
    return (style_image)


def initialize_generated_img(content_image):
    """Initialize generated image by making a noisy picture with slight correlation to
    the original one. This makes the style transfer quicker."""
    generated_image = tf.Variable(
        tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(
        generated_image, clip_value_min=0.0, clip_value_max=1.0)
    return (generated_image)
    
def load_content_style_generated_img(img_size):
    content_image = get_content_img(img_size)
    style_image = get_style_img(img_size)
    generated_image = initialize_generated_img(content_image)
    return content_image, style_image, generated_image
