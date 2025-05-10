import src
import tensorflow as tf
import streamlit as st
from config import settings

def bottom_activations_adversarial(model, img_array):
    # Define a new model that outputs the activations of each layer
    layer_outputs = [layer.output for layer in st.session_state.model.layers]
    activation_model = tf.keras.models.Model(st.session_state.model.get_layer(index=0).input, outputs=layer_outputs)
    # Get the activations of all layers
    activations = activation_model.predict(img_array)

    # Get list of all layer names
    layer_names = [layer.name for layer in model.layers]
    # Streamlit selection box
    default_layer = 'dense'
    default_index = layer_names.index(default_layer) if default_layer in layer_names else 0

 
    selected_layer_name = st.selectbox(
        "Select layer to visualize:", 
        layer_names, 
        index=default_index
    )

    # Define the architecture
    architecture = [
        "conv2d", "max_pooling2d", "conv2d_1", "max_pooling2d_1",
        "conv2d_2", "flatten", "dense", "dense_1"
    ]

    # Highlight the selected layer and grey out the rest
    architecture_display = " → ".join(
        [
            f"<b>{layer}</b>" if layer == selected_layer_name 
            else f"<span style='color:gray'>{layer}</span>" 
            for layer in architecture
        ]
    )
    st.markdown(architecture_display, unsafe_allow_html=True)

    src.visualise_activations_adversarial(activations,activation_model, img_array, selected_layer_name)
        # Show description from the config dictionary
    with st.expander("Layer Description"):
        description = settings.LAYER_DESCRIPTIONS_ADVERSARIAL.get(selected_layer_name, "No description available.")
        st.write(f"**{selected_layer_name}**: {description}")
