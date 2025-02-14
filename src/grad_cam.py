import tensorflow as tf
import numpy as np

# Function to compute Grad-CAM
def grad_cam(model, img_tensor, layer_name):
    # Create a model that gives us both the activations and predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        # Get the activations and predictions from the model
        activations, predictions = grad_model(img_tensor)
        
        # Ensure predictions are in the right shape and get the predicted class index
        print(f"Shape of predictions: {predictions.shape}")
        class_idx = np.argmax(predictions[0])  # Get the class index of the highest prediction (apple/pear)
        class_output = predictions[0][class_idx]  # Access the output corresponding to that class
    
    # Compute the gradient of the class output w.r.t. the activations
    grads = tape.gradient(class_output, activations)
    
    # Compute the pooled gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Reduce over height and width
    
    # Apply the gradients to the activations
    heatmap = activations[0].numpy()
    for i in range(heatmap.shape[-1]):
        heatmap[..., i] *= pooled_grads[i]
    
    # Take the mean of the feature map across all channels
    heatmap = np.mean(heatmap, axis=-1)
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)  # Normalize to [0, 1]
    heatmap = tf.squeeze(heatmap)
    
    return heatmap.numpy()