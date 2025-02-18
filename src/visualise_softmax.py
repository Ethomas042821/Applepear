import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def softmax(logits):
    """Compute softmax values for each class in the logits vector."""
    e_x = np.exp(logits - np.max(logits))  # Numerical stability improvement
    return e_x / e_x.sum(axis=0, keepdims=True)

def calculate_weighted_activations(dense_128_layer_activation, dense_128_weights):
    """
    Calculate the weighted activations for each class using the activations 
    from the Dense(128) layer and the weights to the output layer (Dense(2)).
    
    Args:
        dense_128_layer_activation (np.array): Activations from the Dense(128) layer.
        dense_128_weights (np.array): Weights from Dense(128) to Dense(2) output layer.
        
    Returns:
        np.array: Array with summed weighted activations for each class.
    """
    weighted_activations_sum = np.zeros(2)  # Store sum for both classes (apple = 0, pear = 1)
    
    for i in range(dense_128_layer_activation.shape[0]):  # Loop over all neurons (128 neurons)
        weights_to_output = dense_128_weights[i, :]  # Get the weights from this neuron to output layer

        # Compute the weighted activation for both classes (apple = 0, pear = 1)
        weighted_sum_apple = dense_128_layer_activation[i] * weights_to_output[0]
        weighted_sum_pear = dense_128_layer_activation[i] * weights_to_output[1]

        # Accumulate the weighted activations for each class
        weighted_activations_sum[0] += weighted_sum_apple
        weighted_activations_sum[1] += weighted_sum_pear
    
    # Debugging print for total weighted activations
    print(f"\nTotal Weighted Activations (Apple, Pear): {weighted_activations_sum}")
    return weighted_activations_sum

def visualise_softmax(model, activations):
    """
    Visualize the softmax function and weighted activations for each class.

    Args:
        model (keras.Model): The trained model.
        activations (list): List of layer activations.
    """
    # Extract the activations from the Dense(128) layer (before the output layer)
    dense_128_layer_activation = activations[6][0, :]  # Activations from Dense(128)
    
    # Extract weights from Dense(128) to Dense(2) (output layer) - assuming 2 classes
    dense_128_weights = model.layers[7].get_weights()[0]  # Shape: (128, 2)
    
    # Compute the weighted activations
    weighted_activations_sum = calculate_weighted_activations(dense_128_layer_activation, dense_128_weights)

    # Now we use the weighted activations directly as logits
    logits_range = np.linspace(-10, 10, 100)  # A range of logits to plot the softmax curve
    
    # Apply softmax to each pair of logits
    softmax_values = np.array([softmax(np.array([logit, logit])) for logit in logits_range])  # Softmax for 2 classes

    # Debugging print for softmax probabilities for the weighted activations
    softmax_values_for_weighted_activations = softmax(weighted_activations_sum)
    
    print(f"\nSoftmax for Weighted Activations (Apple): {softmax_values_for_weighted_activations[0]:.4f}")
    print(f"Softmax for Weighted Activations (Pear): {softmax_values_for_weighted_activations[1]:.4f}")

    # Plot the softmax curve
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the weighted activation points on the softmax curve
    ax.plot(weighted_activations_sum[0], softmax_values_for_weighted_activations[0], 'ro', label="Weighted Activation (Apple)", markersize=10)
    ax.plot(weighted_activations_sum[1], softmax_values_for_weighted_activations[1], 'go', label="Weighted Activation (Pear)", markersize=10)

    # Add labels and legend
    ax.set_xlabel("Logits")
    ax.set_ylabel("Softmax Probability")
    ax.set_title("Softmax Curve and Weighted Activations for Each Class")
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

