import streamlit as st
import numpy as np
import page_layout
import tensorflow as tf
from tensorflow.keras.models import load_model

# Display TensorFlow version for informational purposes
# st.write("TensorFlow version: ", tf.__version__)



st.title("The :green[pear]fect :red[apple]")

# Define a function to load the model and apply caching
@st.cache_resource
def load_keras_model():
    try:
        # Attempt to load the Keras model
        model = load_model('applepear_deep.h5')
        return model
    except Exception as e:
        # Handle any errors that may occur during model loading
        st.error(f"Error loading the model: {e}")
        raise e  # Re-raise the exception after logging it

# Load the model into session state if it is not already there
if 'model' not in st.session_state:
    try:
        st.session_state.model = load_keras_model()
    except Exception:
        st.stop()  # Stop execution if the model loading failed

# Access the model from session state
model = st.session_state.model

for layer in model.layers:
    print(layer.name)

# Print all layer names in the model for debugging or informational purposes
# for layer in model.layers:
#     print(layer.name)

st.write("What makes an apple an apple and a pear a pear? I believe it's based on 240,514 parameters. Draw a picture and prove me wrong.")

# Define the layout with two columns
col1, col2 = st.columns(2)

# Left column - drawing canvas
with col1:
    canvas_result = page_layout.column_canvas()

# Right column - display the result
with col2:
    st.header("I think this is...")
    
    # Check if the canvas is completely white (i.e., no drawing made)
    if np.all(canvas_result.image_data == 255):  # Entire canvas is white
        st.warning("Can't wait to see your drawing!")
    else:
        try:
            img_array, img_resized = page_layout.column_prediction(model, canvas_result)
        except Exception as e:
            st.error(f"Error processing the drawing: {e}")

# Toggle button for visualization
on = st.toggle("See your drawing through my eyes!")
if on:
    # Check if drawing was made before attempting predictions
    if np.all(canvas_result.image_data == 255):  # Entire canvas is white
        st.warning("Draw on canvas first!")
    else:
        try:
            page_layout.bottom_activations(model, img_array)
        except Exception as e:
            st.error(f"Error during gradcam and activations display: {e}")



