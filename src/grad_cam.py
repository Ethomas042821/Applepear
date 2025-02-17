import tensorflow as tf
import numpy as np
import streamlit as st

# Function to compute Grad-CAM
def grad_cam(model, img_tensor, layer_name):

    # Ensure eager execution is enabled
    if not tf.executing_eagerly():
        tf.compat.v1.enable_eager_execution()

    # Make sure the layers are trainable
    for layer in st.session_state.model.layers:
         layer.trainable = True

    # Create a model that gives us both the activations and predictions
    conv_layers = [layer.output for layer in st.session_state.model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

    # Get the output of the last convolutional layer
    last_conv_layer_output = conv_layers[-1] if conv_layers else None

    # Get the output of the final layer (output layer)
    output_layer_output = st.session_state.model.layers[-1].output

    
    grad_model = tf.keras.models.Model(
        #inputs=[model.inputs],
        inputs=[st.session_state.model.get_layer(index=0).input],
        #outputs=[model.get_layer(layer_name).output, model.output]
        #outputs=[st.session_state.model.layers[-4].output, output_layer_output]
        outputs = [last_conv_layer_output, output_layer_output]
    )
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)  # Ensure the input image tensor is being watched
        
        # Get the activations and predictions from the model
        st.session_state.activations, predictions = grad_model(img_tensor)
        
        # Debug: Print shape of predictions and activations
        print(f"Predictions shape: {predictions.shape}")
        print(f"Activations shape: {st.session_state.activations[0].shape}")

        print(f"Activations1: {st.session_state.activations[0]}")

        
        class_idx = np.argmax(predictions[0])  # Get the class index of the highest prediction
        print(f"prediction h1gh class idx: {class_idx}")
        st.session_state.class_output = predictions[0][class_idx]  # Access the output corresponding to that class
        print(f"prediction: {st.session_state.class_output}")

    # Compute the gradient of the class output w.r.t. the activations
    grads = tape.gradient(st.session_state.class_output, st.session_state.activations)
    
    # Debug: Check if grads is None
    #if grads is None:
    #    raise ValueError("Gradients are None. Ensure correct class index and layer output.")
    
    # Debug: Print gradient shape
    print(f"Grads shape: {grads.shape}")
    
    # Compute the pooled gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Reduce over height and width
    pooled_grads = pooled_grads / tf.norm(pooled_grads)  # Normalize the pooled gradients in order to better see their impact
    
    # Apply the gradients to the activations
    heatmap = st.session_state.activations[0].numpy()
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