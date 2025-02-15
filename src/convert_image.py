import numpy as np
from PIL import Image

def convert_image(img):
    # Convert the drawn image data into a PIL Image
    img = img.convert('L')  # Convert image to grayscale (L mode)
    img_resized = img.resize((28, 28))  # Resize to 28x28

    # Convert the image to a numpy array
    img_array = np.array(img_resized)

    # Invert the image: subtract from 255
    #img_array = 255 - img_array

    # Rescale the image for normalization
    img_array = img_array / 255.0

    # Add a batch dimension (model expects a batch of images)
    img_array = np.expand_dims(img_array, axis=-1)  # Shape (28, 28, 1)

    # Add another dimension for the batch
    img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 28, 28, 1)

    return img_resized,img_array