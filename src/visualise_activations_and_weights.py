import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def visualise_activations_and_weights(model, activations, img_array, top_k=5):
    # Get the layer names from the model, excluding the input layer (if any)
    layer_names = [layer.name for layer in model.layers if 'input' not in layer.name]

    # Extract activations from Dense(128) layer (before the output layer)
    dense_128_layer_activation = activations[6][0, :]  # Get activations from Dense(128)

    # Get the top `k` activations and their indices
    top_k_indices = np.argsort(dense_128_layer_activation)[-top_k:][::-1]
    top_k_activations = dense_128_layer_activation[top_k_indices]

    # Get the corresponding weights from Dense(128) to Dense(2) output layer
    dense_128_weights = model.layers[7].get_weights()[0]  # Weights from Dense(128) -> Dense(2)
    #print(dense_128_weights)

        # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot activations on the right axis (ax1)
    ax1.bar(range(top_k), top_k_activations, color=plt.cm.viridis(top_k_activations / max(top_k_activations)), label="Top k Activations")
    ax1.set_ylabel("Activation Value", color='black')
    ax1.set_xlabel("Neurons (Top k)")
    ax1.set_title(f"Top {top_k} Activations and Weights")
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second y-axis to plot the weights on the left side
    ax2 = ax1.twinx()  # Create a second axis that shares the same x-axis

    # List to store results for highlighting
    highlight_indices = []
    # Plot corresponding weights to the output layer (Dense(2)) on ax2
    offset = 0.2  # Set a small offset between neurons' weights (to avoid overlap)
    for i, idx in enumerate(top_k_indices):
        weights_to_output = dense_128_weights[idx, :]  # Weights to the two output neurons (apple, pear)

        # Multiply activations by corresponding weights
        weighted_sum_apple = top_k_activations[i] * weights_to_output[0]
        weighted_sum_pear = top_k_activations[i] * weights_to_output[1]

        if (st.session_state.class_idx == 0 and weighted_sum_apple> weighted_sum_pear):
            highlight_indices.append(i)

        if (st.session_state.class_idx == 1 and weighted_sum_apple < weighted_sum_pear):
            highlight_indices.append(i)
        #I wanted to show pointing arrows up and down, but currently i use an "8" markerbfor both because its less confusing in the end
        if(weights_to_output[0] <= 0):
            # Plot the weight to the "apple" class (class 0) on the left axis (ax2)
            ax2.plot(i - offset, weights_to_output[0], 'r8', label="Weight to class Apple" if i == 0 else "")
        if(weights_to_output[0] > 0):
            ax2.plot(i - offset, weights_to_output[0], 'r8', label="Weight to class Apple" if i == 0 else "")
        if(weights_to_output[1] <= 0):
            # Plot the weight to the "pear" class (class 1) on the left axis (ax2)
            ax2.plot(i + offset, weights_to_output[1], 'g8', label="Weight to class Pear" if i == 0 else "")
        if(weights_to_output[1] > 0):
            ax2.plot(i + offset, weights_to_output[1], 'g8', label="Weight to class Pear" if i == 0 else "")


    # Add a dotted line at y = 0 to represent the threshold
    #ax2.axhline(y=0, color='black', linestyle='--', label="Weight = 0")

    # Highlight the bars for the neurons where the prediction is correct
    for idx in highlight_indices:
        ax1.bar(idx, top_k_activations[idx], color="None", edgecolor='black', linewidth=3) 

    # Set the left axis label (for weights)
    ax2.set_ylabel("Weight Value", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Show legend for weights
    ax2.legend(loc='upper left')

    # Show the plot
    st.pyplot(fig)

