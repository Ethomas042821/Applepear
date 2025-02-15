import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import src
import page_layout

st.title("The :green[pear]fect :red[apple]")
# Load the model
model = load_model('applepear_invert_nolastlayer.h5')

for layer in model.layers:
    print(layer.name)

st.write("What makes an apple an apple and a pear a pear? I believe it's based on 203,394 parameters. Draw a picture and prove me wrong.")
# Define your columns for layout
col1, col2 = st.columns(2)

# Left column - drawing canvas
with col1:
    canvas_result = page_layout.column_canvas()

# Right column - display the resized image after clicking the button
with col2:
    #img_array, img_resized = page_layout.column_prediction(model, canvas_result)
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
        st.write("Hello! I'm a convolutional neural network, trained to recognize apples and pears. I’m pretty good at it, if I do say so myself. Am I smart? I’ve been trained on over 144,000 apple images and nearly 117,000 pear images. How many examples would YOU need to see to tell them apart? 10? 5? 1?")
        st.write('Apples and pears are pretty much all I know. So, if you draw anything else, I’ll tell you at least if it’s more apple-like (let’s call that "applish" just for fun) or more pear-like ("pearish").')
        st.write('But what exactly do "applish" and "pearish" mean? Let’s take a look at your image through my eyes and I’ll show you how I see it.')
        st.write("The heatmap highlights the parts of the image that matter most to me when making my decision. If I recognize your drawing as an apple, the red areas show the most 'applish' parts of the apple sketch. If it’s a pear, the red areas highlight the most 'pearish' parts.")
        # Convert the image to a TensorFlow tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        # Compute Grad-CAM heatmap
        heatmap = src.grad_cam(model, img_tensor, layer_name='conv2d_1')  # last conv layer
        # Convert the input image tensor to a numpy array for plotting
        img = img_tensor[0].numpy()
        # Plot the result with the heatmap overlaid
        plothtmp = src.plot_grad_cam(img, heatmap)
        st.image(plothtmp,width=192)
        st.write('Why are broad regions highlighted instead of just the parts of the lines in the sketch? Let me show you how I process the image to explain:')
        # Display the resized image
        st.image(img_resized, caption="28x28 Resized Image", width = 192)
        # Define a new model that outputs the activations of each layer
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
        # Get the activations of all layers
        activations = activation_model.predict(img_array)
        # Visualize activations for each layer in Streamlit
        st.write("I don’t see a sketch—I see an array of numbers between 0 and 1 (grayscale pixels). My algorithm goes through several steps (layers) to detect key patterns (features), which I’ve been trained to examine. Let me show you exactly how I process your sketch at each layer.")
        src.visualise_activations(activations,activation_model)

