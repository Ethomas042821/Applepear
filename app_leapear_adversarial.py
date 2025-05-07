import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import page_layout
import src  # Assuming src is a module in the same directory

import sys
print(sys.executable)

# Display TensorFlow version for informational purposes
# st.write("TensorFlow version: ", tf.__version__)

st.title("The :green[pear]fect :red[apple] under :blue[attack]!")

# Define a function to load the model and apply caching
@st.cache_resource
def load_keras_model():
    try:
        # Attempt to load the Keras model
        model = load_model('applepear_deep.h5')  # Your model file
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

# Print all layer names in the model for debugging or informational purposes
# for layer in model.layers:
#     print(layer.name)

st.write("What makes an apple an apple and a pear a pear? I believe it's based on 240,514 parameters.")
st.write("...But maybe its just a few additional pixels after all. Let's find out!")

with st.expander("How does it work?"):
    st.write("""
    I am a Convolutional Neural Network (CNN) trained to classify images of apples and pears. I use a deep learning model with multiple layers to extract features from the images and make predictions.
    
    You can draw an apple or a pear on the left canvas, and I will try to classify it. You can also add some noise to the image to see how I react to it.
    
    The adversarial attack used here is the **Fast Gradient Sign Method (FGSM)**. It's a simple but powerful technique to fool neural networks by adding a small amount of noise in the direction that maximally increases the model's loss.
    """)

    st.latex(r"""
    x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
    """)

    st.write("""
    - \( x \): original input image  
    - \( \epsilon \): small scalar controlling the noise level  
    - \( \nabla_x J(\theta, x, y) \): gradient of the loss with respect to the input  
    - \( x_{\text{adv}} \): adversarial image  
    """)

# Define the layout with two columns
col1, col2, col3 = st.columns(3)

# Left column - drawing canvas and adversarial image
with col1:
    canvas_result = page_layout.column_canvas_adversarial()

with col2:
    st.header("Model input:")
    epsilon = st.slider(":blue[Adversarial Noise Level ($\epsilon)]", 0.0, 0.3, 0.0, 0.005)

# Right column - display the result
with col3:
    st.header("I think this is...")

    # Check if the canvas is completely white (i.e., no drawing made)
    if np.all(canvas_result.image_data == 255):  # Entire canvas is white
        st.warning("Can't wait to see your drawing!")
    else:
        try:

            # Get preprocessed input and resized image
            img_array, img_resized = page_layout.just_retrieve_image(canvas_result)

            # # Make prediction on original image
            pred = model.predict(img_array)
            label = np.argmax(pred)
            confidence = np.max(pred)

            class_names = ['Apple', 'Pear']  # Adjust to match your model
            # Generate adversarial pattern
            perturbations = src.create_adversarial_pattern(model, img_array, [label])

            # Apply adversarial noise
            adversarial = img_array + epsilon * perturbations
            adversarial = tf.clip_by_value(adversarial, 0, 1)

            # Predict again on adversarial image
            pred_adv = model.predict(adversarial)
            label_adv = np.argmax(pred_adv)
            confidence_adv = np.max(pred_adv)


            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                # Display adversarial image
                st.image(adversarial.numpy(), use_container_width=False,width = 100)
                    # Add slider to control adversarial noise
                

            page_layout.adversarial_column_prediction(model, adversarial)

            # # Original prediction display
            # st.markdown(f"**Original Prediction:** {class_names[label]} ({confidence * 100:.2f}%)")

            # # Adversarial prediction display
            # st.markdown(f"**Adversarial Prediction:** {class_names[label_adv]} ({confidence_adv * 100:.2f}%)")

        except Exception as e:
            st.error(f"Error processing the drawing: {e}")

with st.expander("Hint"):
    st.write("Smaller apples and pears are easier to fool. Try drawing a small one!")
st.markdown("<br>", unsafe_allow_html=True)
st.caption("If you want to see what exactly happens in each layer of the model, check out this interactive app: applepear.streamlit.app")