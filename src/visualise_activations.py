import streamlit as st
import matplotlib.pyplot as plt
from config import settings
import src

def visualise_activations(activations, model,img_array, num_columns=8):
    # Get the layer names from the model, excluding the input layer (if any)
    layer_names = [layer.name for layer in model.layers if 'input' not in layer.name]

    # Check if the number of activations and layer names match
    assert len(layer_names) == len(activations), "Number of layers and activations don't match!"

    # Loop through the activations and layer names to display the results
    for i, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
        # Get the step and description for the current layer
        if layer_name in settings.layer_info:
            step = settings.layer_info[layer_name]["step"]
            description = settings.layer_info[layer_name]["description"]
        else:
            step = f"Step {i+1}: {layer_name}"  # Default step if layer not in layer_info
            description = "No description available for this layer."  # Default description if layer not in layer_info
            
        # Display the step (layer name with step)
        st.subheader(step)  # Display step as a subheader
        
        # Display the description for the current layer
        st.markdown(description)  # Display description as a Markdown block

        # Check the shape of the activation and handle accordingly
        if len(layer_activation.shape) == 4:  # Convolutional layer (batch_size, height, width, num_filters)
            num_filters = layer_activation.shape[-1]  # Number of filters (channels)
            size = layer_activation.shape[1]  # Height and width of the feature maps

            # Number of rows needed to display all filters
            num_rows = num_filters // num_columns
            if num_filters % num_columns != 0:
                num_rows += 1

            # Plotting the feature maps (activations)
            fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns*2, num_rows*2))
            axes = axes.flatten()

            for j in range(num_filters):
                ax = axes[j]
                ax.imshow(layer_activation[0, :, :, j], cmap='viridis')  # Display one filter's feature map
                ax.axis('off')  # Turn off axes

            # Display the plot in Streamlit
            st.pyplot(fig)

        elif len(layer_activation.shape) == 2:  # Flatten or Dense layer (batch_size, num_units)

            if layer_name == 'dense_1':#the last layer before uotput
                st.write(f"""
                         In the final layer of the network, the decision is made.  For each class—apple and pear—I calculate a score. This score comes from multiplying the activation (or brightness) of each neuron by its weight for that class. Then, I sum up all these results for each class. The class that has the higher score becomes my final prediction.
                        \n
                        But wait—let's take a closer look.
                        \n    
                        We'll examine the five most active neurons from the previous layer. These are the neurons that have the greatest impact on the network's decision. For each of these neurons, I will show you how the activations and corresponding weights contribute to the {st.session_state.predicted_class_for_desc} classification of your sketch.
                         
                         """)
                src.visualise_activations_and_weights(st.session_state.model, activations, img_array)
                st.write(f"""
                            In this visualization, each neuron is represented as a bar with heigh corresponding to its activation. 
                            \n
                            For every neuron, I display two corresponding weights—one for the apple class and one for the pear class. 
                            \n
                             The strength of a neuron's contribution to the predicted class is determined by multiplying its activation by the weight for that class. This gives us a measure of how much each neuron influences the final decision for either apple or pear.
                            \n
                            The neurons that have the most influence on predicting {st.session_state.predicted_class_for_desc} (compared to {st.session_state.unpredicted_class_for_desc} ) are highlighted with a black frame. (Sometimes, this will be all five neurons in the plot.)
                            \n
                            That’s really the crux of it. When I say '{st.session_state.fun_predicted_class_for_desc}', what I mean is this: a '{st.session_state.fun_predicted_class_for_desc}' sketch is one that activates neurons which favor the {st.session_state.predicted_class_for_desc} class, rather than the {st.session_state.unpredicted_class_for_desc} class. 
                            \n
                            The final output layer has only two neurons, one for apple( left), one for pear (right). The one with highest score shines:
                            """)
            # Visualize 2D activations (fully connected layers)
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.imshow(layer_activation[0, :].reshape(1, -1), cmap='viridis', aspect='auto')
            ax.axis('off')  # Turn off axes
            st.pyplot(fig)


           
            # if layer_name == 'dense_1':#the output_layer
            #     src.visualise_softmax(st.session_state.model, activations) #this is even more confusing when trying to visualise so i skip it for now