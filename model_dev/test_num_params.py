
from tensorflow.keras import layers, models


# Build the CNN model
model = models.Sequential()
# Add the first convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))
model.add(layers.MaxPooling2D((2, 2)))  # MaxPooling to reduce the spatial dimensions
# Add a second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Add a third convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
# Flatten the output of the convolutional layers
model.add(layers.Flatten())
# Add a dense (fully connected) layer
model.add(layers.Dense(128, activation='relu'))
# Add Dropout layer with 0.5 (50%) probability
#model.add(layers.Dropout(0.5))
# Output layer 
model.add(layers.Dense(2, activation='softmax'))  

# Compile the model
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',  # Use this if your labels are integers
               metrics=['accuracy'])

print("model compiled")

model.summary()