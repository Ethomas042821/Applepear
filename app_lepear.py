import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import src

st.title("The :green[pear]fect :red[apple]")
# Load the model
model = load_model('applepear.h5')

for layer in model.layers:
    print(layer.name)

st.write("What makes an apple an apple and a pear a pear? I believe it's based on 203,394 parameters. Draw a picture and prove me wrong.")
# Define your columns for layout
col1, col2 = st.columns(2)

# Left column - drawing canvas
with col1:
    st.header("Draw Here")
    st.write("(apple or pear)")  
    canvas_result = st_canvas(
        fill_color="white",  # Set the background color
        stroke_width=15,  # Set the stroke width
        stroke_color="black",  # Set the stroke color
        background_color="white",  # Background color
        width=192,  # Width of the canvas
        height=192,  # Height of the canvas
        drawing_mode="freedraw",  # Drawing mode
        key="canvas",  # Key to access the canvas state
    )


# Right column - display the resized image after clicking the button
with col2:
    st.header("I think this is...")
    if canvas_result.image_data is not None:
        # Check if the image is not all white (empty canvas)
        if np.all(canvas_result.image_data == 255):  # If the entire canvas is white (255 for grayscale)
            st.warning("Cant wait to see your drawing!")
        else:
            #Load image from canvas
            img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
            #Prepare image for the model
            img_resized,img_array = src.convert_image(img)
            # Make prediction
            prediction = model.predict(img_array)
            # Flatten the prediction array to remove the first dimension (batch size of 1)
            prediction = prediction.flatten()  # This will make the array shape (2,)

            # Define class labels for answer
            class_labels_color = ['...an :red[apple]!','...a :green[pear]!']
            # Get the index of the class with the highest probability
            class_idx = np.argmax(prediction)  # This returns the index of the highest value (0 or 1)
            # Get the predicted class label
            predicted_class = class_labels_color[class_idx]
            # Display the prediction and confidence
            st.header(f"{predicted_class}")

            # Define class labels for piechart
            class_labels = ['apple','pear']
            piechart = src.create_piechart(prediction, class_labels)
            # Display the pie chart
            st.pyplot(piechart)
    else:
        st.warning("Cant wait to see your drawing!")

on = st.toggle("See your drawing through my eyes")
if on:
    if np.all(canvas_result.image_data == 255):  # If the entire canvas is white (255 for grayscale)
            st.warning("Draw on canvas first!")
    else:
       
        st.write("The heatmap shows, which part of the image are important to me to decide what you drew. In other words, if I recognized your sketch as an apple red regions show the most 'applish' parts of apple sketch. Or the most 'pearisch' parts of a pear sketch in the other case.")
        # Convert the image to a TensorFlow tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        # Compute Grad-CAM heatmap
        heatmap = src.grad_cam(model, img_tensor, layer_name='conv2d_2')  # last conv layer
        # Convert the input image tensor to a numpy array for plotting
        img = img_tensor[0].numpy()
        # Plot the result with the heatmap overlaid
        plothtmp = src.plot_grad_cam(img, heatmap)
        st.image(plothtmp,width=192)
        st.write('Why are broad regions highlighted, not just the sketch parts? To  answer this question, lets look how I process the image:')
        # Display the resized image
        st.image(img_resized, caption="28x28 Resized Image", width = 192)
        # Define a new model that outputs the activations of each layer
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
        # Get the activations of all layers
        activations = activation_model.predict(img_array)
        # Visualize activations for each layer in Streamlit
        st.write("I dont see a sketch. I see an array of numbers between 0 and 1 (grayscale pixels). My algorythm will apply a few steps (layers) to detect important patterns, which allows me to identify features that I am trained to recognize. Let me show you exactly how I process your sketch in each layer")
        src.visualise_activations(activations,activation_model)

