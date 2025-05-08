import tensorflow as tf


def compute_saliency_map(model, img_array, class_index):
    img_array = tf.convert_to_tensor(img_array)
    img_array = tf.cast(img_array, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        predictions = model(img_array)
        loss = predictions[:, class_index]

    gradients = tape.gradient(loss, img_array)
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    saliency = tf.squeeze(saliency)  # Remove batch dimension
    return saliency.numpy()
