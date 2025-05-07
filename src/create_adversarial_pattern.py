import tensorflow as tf

def create_adversarial_pattern(model, input_image, input_label):
    # Convert input to a tensor with float32 dtype and ensure it has gradient tracking
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

    # Use GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as tape:
        tape.watch(input_image)  # Explicitly watch the input
        prediction = model(input_image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)

    # Get the gradients of the loss with respect to the input image
    gradient = tape.gradient(loss, input_image)

    # Get the sign of the gradients to create the adversarial pattern
    signed_grad = tf.sign(gradient)

    return signed_grad