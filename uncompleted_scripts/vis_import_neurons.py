import tensorflow as tf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Function to calculate gradients for each neuron in the dense layer (with positive gradients only)
def calculate_positive_neuron_gradients(activations, model, img_tensor, output_class_idx):
    # Compute gradients of the class output with respect to the dense layer activations
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)  # Ensure the input image tensor is watched
        output = model(img_tensor)  # Get the model's output
        class_output = output[0, output_class_idx]  # Select the class output

    # Get the gradients of the class output with respect to the dense activations
    grads = tape.gradient(class_output, activations)
    
    # Pool the gradients (mean over the neurons)
    pooled_grads = tf.reduce_mean(grads, axis=0)

    # Create a dictionary of neurons with their respective gradients (only positive gradients)
    neuron_gradients = {i: abs(pooled_grads[i].numpy()) for i in range(len(pooled_grads)) if pooled_grads[i] > 0}
    
    # Sort neurons based on their gradient magnitudes (highest to lowest)
    sorted_neurons = sorted(neuron_gradients.items(), key=lambda x: x[1], reverse=True)

    return sorted_neurons, pooled_grads

# Function to visualize the activations with only positive gradients in the dense layer
def visualize_activation_with_positive_gradients(activations, model, img_tensor, output_class_idx, top_n=10):
    # Get the sorted list of neurons based on their positive gradients
    sorted_neurons, pooled_grads = calculate_positive_neuron_gradients(activations, model, img_tensor, output_class_idx)

    # Create a new activation map where neurons with negative gradients are zeroed out
    filtered_activation = activations.copy()  # Copy the original activations

    # Zero out the activations for neurons with negative gradients
    for i in range(len(pooled_grads)):
        if pooled_grads[i] <= 0:  # If the gradient is negative
            filtered_activation[0, i] = 0  # Set the activation of that neuron to zero

    # Visualize the filtered activation map (only neurons with positive gradients)
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(filtered_activation[0, :].reshape(1, -1), cmap='viridis', aspect='auto')
    ax.axis('off')  # Turn off axes
    st.pyplot(fig)

    # Optionally, display the top N neurons with positive gradients
    st.write(f"Top {top_n} most important neurons (with positive gradients) for class {output_class_idx}:")
    for i, (neuron_idx, gradient_magnitude) in enumerate(sorted_neurons[:top_n]):
        st.write(f"Neuron {neuron_idx}: Gradient magnitude = {gradient_magnitude}")