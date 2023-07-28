from neural_style_transfer.cost import compute_content_cost, compute_style_cost, total_cost
from neural_style_transfer.utils import clip_0_1, tensor_to_image
import tensorflow as tf
from neural_style_transfer.loader import load_content_style_generated_img
from neural_style_transfer.model import style_image_encoding, content_image_encoding,get_vgg_output_model

def train():
    # Show the generated image at some epochs
    # Uncomment to reset the style transfer process. You will need to compile the train_step function again 
    epochs = 50
    img_size = 400
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    output_model, style_layers = get_vgg_output_model(img_size)
    content_image, style_image, generated_image = load_content_style_generated_img(img_size)
    content_encoder, style_encoder = content_image_encoding(output_model, content_image), style_image_encoding(output_model, style_image)
    a_C, a_S = content_encoder, style_encoder

    @tf.function()
    def train_step(generated_image):
        with tf.GradientTape() as tape:
            # In this function you must use the precomputed encoded images a_S and a_C
            
            a_G = output_model(generated_image)
            
            J_style = compute_style_cost(a_S, a_G, style_layers)

            # Compute the content cost
            J_content = compute_content_cost(a_C, a_G)
            # Compute the total cost
            J = total_cost(J_content, J_style, alpha = 10, beta = 40)
            
        grad = tape.gradient(J, generated_image)

        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(clip_0_1(generated_image))
        # For grading purposes
        return J

    for i in range(epochs):
        train_step(generated_image)
        if i % 50 == 0:
            print(f"Epoch {i} ")
        if i % 50 == 0:
            image = tensor_to_image(generated_image)
            image.save(f"output/image_{i}.jpg")