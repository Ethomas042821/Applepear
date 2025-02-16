import tensorflow as tf
import numpy as np

# Function to compute Grad-CAM
def grad_cam(model, img_tensor, layer_name):
    #inputs = tf.keras.Input(shape=(28, 28, 1))
    #outputs = model(inputs)
    # Create a model that gives us both the activations and predictions
    # Ensure the model's last convolutional layer is passed
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)  # Ensure the input image tensor is being watched
        
        # Get the activations and predictions from the model
        activations, predictions = grad_model(img_tensor)
        
        # Debug: Print shape of predictions and activations
        print(f"Predictions shape: {predictions.shape}")
        print(f"Activations shape: {activations[0].shape}")
        
        class_idx = np.argmax(predictions[0])  # Get the class index of the highest prediction
        class_output = predictions[0][class_idx]  # Access the output corresponding to that class

    # Compute the gradient of the class output w.r.t. the activations
    grads = tape.gradient(class_output, activations)
    
    # Debug: Check if grads is None
    if grads is None:
        raise ValueError("Gradients are None. Ensure correct class index and layer output.")
    
    # Debug: Print gradient shape
    print(f"Grads shape: {grads.shape}")
    
    # Compute the pooled gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Reduce over height and width
    pooled_grads = pooled_grads / tf.norm(pooled_grads)  # Normalize the pooled gradients in order to better see their impact
    
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

    #print(heatmap.numpy())
    
    return heatmap.numpy()