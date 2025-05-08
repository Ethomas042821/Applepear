import src
import tensorflow as tf
import streamlit as st

def bottom_activations_adversarial(model, img_array):
    # Define a new model that outputs the activations of each layer
    layer_outputs = [layer.output for layer in st.session_state.model.layers]
    activation_model = tf.keras.models.Model(st.session_state.model.get_layer(index=0).input, outputs=layer_outputs)
    # Get the activations of all layers
    activations = activation_model.predict(img_array)
    src.visualise_activations_adversarial(activations,activation_model, img_array)
