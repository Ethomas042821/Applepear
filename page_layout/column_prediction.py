import streamlit as st
from PIL import Image
import numpy as np
import src


def column_prediction(model, canvas_result):
    # Check if canvas_result or its image_data is None
    if canvas_result is None or canvas_result.image_data is None:
        st.error("No image data received from the canvas.")
        return None, None
    
    try:
        # Load image from canvas
        img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
    except Exception as e:
        st.error(f"Error converting canvas data to image: {e}")
        return None, None

    # Prepare image for the model 
    img_resized, img_array = src.convert_image(img)
    
    if img_array is None:
        st.error("Image array is None. Check the conversion process.")
        return None, None

    # Make prediction using the model
    try:
        prediction = model.predict(img_array)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None, None

    # Flatten the prediction array to remove the first dimension (batch size of 1)
    prediction = prediction.flatten()  # This will make the array shape (2,)

    # Define class labels for answer (for the header)
    class_labels_color = ['...an :red[apple]!','...a :green[pear]!']
    
    # Ensure the prediction array is valid
    if len(prediction) != 2:
        st.error(f"Prediction array has an unexpected shape: {prediction.shape}. Expected shape (2,).")
        return None, None

    # Get the index of the class with the highest probability
    class_idx = np.argmax(prediction)  # This returns the index of the highest value (0 or 1)
    not_sel_class_idx = np.argmin(prediction)  # This returns the index of the lowest value (0 or 1)

    st.session_state.class_idx = class_idx
    
    # Get the predicted class label
    predicted_class = class_labels_color[class_idx]
    
    # Display the prediction and confidence
    st.header(f"{predicted_class}")

    # Define class labels for the piechart (for visualization)
    class_labels = ['apple', 'pear']
    class_fun_labels = ['applish', 'pearish']

    # Get the predicted class label
    st.session_state.predicted_class_for_desc = class_labels[class_idx]
    st.session_state.unpredicted_class_for_desc = class_labels[not_sel_class_idx]

    st.session_state.fun_predicted_class_for_desc = class_fun_labels[class_idx]
    st.session_state.fun_unpredicted_class_for_desc = class_fun_labels[not_sel_class_idx]
    
    # Create and display the pie chart
    piechart = src.create_piechart(prediction, class_labels)
    st.pyplot(piechart)

    return img_array, img_resized


