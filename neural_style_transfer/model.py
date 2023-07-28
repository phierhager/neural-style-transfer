import tensorflow as tf
from neural_style_transfer.loader import load_vgg


def get_vgg_style_layers():
    """Depends upon used model."""
    STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)]
    return STYLE_LAYERS

def get_vgg_content_layer():
    return [('block5_conv4', 1)]

def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def get_vgg_output_model(img_size):
    style_layers = get_vgg_style_layers()
    model = get_layer_outputs(load_vgg(img_size), style_layers + get_vgg_content_layer())
    return model, style_layers

def content_image_encoding(output_model, content_image):
    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = output_model(preprocessed_content)
    return a_C
    
def style_image_encoding(output_model, style_image):
    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = output_model(preprocessed_content)
    return a_S
