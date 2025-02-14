import streamlit as st
import matplotlib.pyplot as plt
from config import settings


# Function to visualize activations (feature maps)
def visualise_activations(activations,model, num_columns=8):
    # Get the layer names from the model
    layer_names = [layer.name for layer in model.layers]

    # Get the layer names from the model, excluding the input layer (if any)
    layer_names = [layer.name for layer in model.layers if 'input' not in layer.name]

    # Check if the number of activations and layer names match
    assert len(layer_names) == len(activations), "Number of layers and activations don't match!"

    # Loop through the activations and layer names to display the results
    for i, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
        # Display the description for the current layer
        st.write(settings.layer_descriptions.get(layer_name, "No description available for this layer."))

        # Check the shape of the activation and handle accordingly
        if len(layer_activation.shape) == 4:  # Convolutional layer (batch_size, height, width, num_filters)
            num_filters = layer_activation.shape[-1]  # Number of filters (channels)
            size = layer_activation.shape[1]  # Height and width of the feature maps

            # Number of rows needed to display all filters
            num_rows = num_filters // num_columns
            if num_filters % num_columns != 0:
                num_rows += 1

            # Plotting the feature maps (activations)
            fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns*2, num_rows*2))
            axes = axes.flatten()

            for j in range(num_filters):
                ax = axes[j]
                ax.imshow(layer_activation[0, :, :, j], cmap='viridis')  # Display one filter's feature map
                ax.axis('off')  # Turn off axes

            # Display the plot in Streamlit
            st.pyplot(fig)

        elif len(layer_activation.shape) == 2:  # Flatten or Dense layer (batch_size, num_units)
            # Visualize 2D activations (often used for fully connected layers)
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.imshow(layer_activation[0, :].reshape(1, -1), cmap='viridis', aspect='auto')
            ax.axis('off')  # Turn off axes
            st.pyplot(fig)