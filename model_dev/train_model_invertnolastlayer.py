import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load the .npz file
data = np.load(r'C:\Users\kmoud\Documents\projekty\ApplePear\data\combined_applepear_dataset_2.npz')

# Inspect the content of the .npz file to check the keys and understand the structure
print("Keys in the .npz file:", data.files)  # Check the keys (e.g., 'images', 'labels')
print("Shape of images:", data['images'].shape)  # Check the shape of the images
print("Shape of labels:", data['labels'].shape)  # Check the shape of the labels

# Check a few values in the labels to see if they are integers or one-hot encoded
print("Unique labels values:", np.unique(data['labels']))  # Unique values in labels
print("Labels example (first 5):", data['labels'][:5])  # Show first 5 labels

# Step 2: Extract images and labels (i start with 5000 rows)
not_inverted_images = data['images']
labels = data['labels']
# labels = data['labels'][:500000]

images = 255 - not_inverted_images

apple_images = images[labels == 0]  # Apple class (label 0)
pear_images = images[labels == 1]   # Pear class (label 1)

# Get the number of apple and pear images
num_apple_images = len(apple_images)
num_pear_images = len(pear_images)

# Display the counts
st.write(f'Number of apple images: {num_apple_images}')
st.write(f'Number of pear images: {num_pear_images}')

# Set up the figure for displaying images
fig, axes = plt.subplots(2, 10, figsize=(15, 5))

# Display 5 random apple images
for i in range(10):
    axes[0, i].imshow(apple_images[i], cmap='gray')  # Display in grayscale
    axes[0, i].axis('off')  # Hide axes for better presentation
    axes[0, i].set_title(f"Apple {i+1}")

# Display 5 random pear images
for i in range(10):
    axes[1, i].imshow(pear_images[i], cmap='gray')  # Display in grayscale
    axes[1, i].axis('off')  # Hide axes for better presentation
    axes[1, i].set_title(f"Pear {i+1}")

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot as a JPG file
fig.savefig('image_examples.jpg', format='jpg')

# Show the plot in Streamlit
st.pyplot(fig)

# Normalize the images to be between 0 and 1 (if not already binary)
images = images.astype('float32') / 255.0  # Normalize the image pixel values to [0, 1]
print("images normalized")

# Ensure the images have a channel dimension 
if images.ndim == 3:  # images should be (num_samples, height, width)
    images = np.expand_dims(images, axis=-1)  # Add a channel dimension to make it (num_samples, height, width, 1)
print("images chanelled")

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
print("traintestsplit done")

# Build the CNN model
model = models.Sequential()
# Add the first convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], images.shape[3])))
model.add(layers.MaxPooling2D((2, 2)))  # MaxPooling to reduce the spatial dimensions
# Add a second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Flatten the output of the convolutional layers
model.add(layers.Flatten())
# Add a dense (fully connected) layer
model.add(layers.Dense(64, activation='relu'))
# Output layer 
model.add(layers.Dense(2, activation='softmax'))  

# Compile the model
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',  # Use this if your labels are integers
               metrics=['accuracy'])

print("model compiled")

# Train the CNN model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

print("model fitted")

# Save the model to a file 
model.save('applepear_invert_nolastlayer.h5')  
print("model saved")

# Plot training and validation accuracy
fig_acc, ax_acc = plt.subplots()
ax_acc.plot(history.history['accuracy'], label='Training Accuracy')
ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax_acc.set_xlabel('Epochs')
ax_acc.set_ylabel('Accuracy')
ax_acc.set_title('Training and Validation Accuracy')
ax_acc.legend()

# Save the accuracy plot as JPG
fig_acc.savefig('accuracy_plot_invert_nolastlayer.jpg', format='jpg')

# Plot training and validation loss
fig_loss, ax_loss = plt.subplots()
ax_loss.plot(history.history['loss'], label='Training Loss')
ax_loss.plot(history.history['val_loss'], label='Validation Loss')
ax_loss.set_xlabel('Epochs')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Training and Validation Loss')
ax_loss.legend()

# Save the loss plot as JPG
fig_loss.savefig('loss_plot_invert_nolastlayer.jpg', format='jpg')

# Show both plots in Streamlit
st.pyplot(fig_acc)
st.pyplot(fig_loss)