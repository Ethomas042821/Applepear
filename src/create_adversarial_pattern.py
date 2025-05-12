import tensorflow as tf
import numpy as np

def create_adversarial_pattern(model, input_image):
    # Convert input to a tensor with float32 dtype and ensure it has gradient tracking
                # # Make prediction on original image
    print("before model predict")
    pred = model.predict(np.array(input_image))
    print("after model predict")
    input_label = tf.convert_to_tensor([np.argmax(pred)], dtype=tf.int32)
    print("after input label")

    input_image = tf.convert_to_tensor(np.array(input_image), dtype=tf.float32)
    print("after convert to tensorn")

    # Use GradientTape to record operations for automatic differentiation
    with tf.GradientTape() as tape:
        tape.watch(input_image)  # Explicitly watch the input
        print("after input image")
        prediction = model(input_image)
        print("after prediction = model(input_image)")
        loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)
        print("after loss")
    # Get the gradients of the loss with respect to the input image
    gradient = tape.gradient(loss, input_image)
    print("after gradient")
    # Get the sign of the gradients to create the adversarial pattern
    signed_grad = tf.sign(gradient)
    print("after signum")

    return signed_grad
