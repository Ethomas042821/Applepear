import streamlit as st
from PIL import Image
import numpy as np
import src


def column_prediction(model, canvas_result):
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
            
            return img_array, img_resized

