import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import page_layout

st.title("The :green[pear]fect :red[apple]")

# Define a function to load the model and apply caching
@st.cache_resource
def load_keras_model():
    # Load the Keras model
    model = load_model('applepear_deep.h5')
    return model

# Call the function to load the model
model = load_keras_model()

for layer in model.layers:
    print(layer.name)

st.write("What makes an apple an apple and a pear a pear? I believe it's based on 240,514 parameters. Draw a picture and prove me wrong.")

col1, col2 = st.columns(2)
# Left column - drawing canvas
with col1:
    canvas_result = page_layout.column_canvas()

# Right column - display the result
with col2:
    #
    st.header("I think this is...")
    if np.all(canvas_result.image_data == 255):  # If the entire canvas is white (255 for grayscale)
        st.warning("Cant wait to see your drawing!")
    else:    
        img_array, img_resized = page_layout.column_prediction(model, canvas_result)
   
on = st.toggle("See your drawing through my eyes!")
if on:
    if np.all(canvas_result.image_data == 255):  # If the entire canvas is white (255 for grayscale)
            st.warning("Draw on canvas first!")
    else:
        page_layout.bottom_activations(model, img_array)

