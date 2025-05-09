import src
import tensorflow as tf
import streamlit as st

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

    src.visualise_activations_adversarial(activations,activation_model, img_array, selected_layer_name)
