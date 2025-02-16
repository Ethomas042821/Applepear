layer_info = {
    "conv2d": {
        "step": "Convolutional Layer 1",
        "description": "This is the first convolutional layer, which scans the input sketch with small filters to detect basic features. In this case, there are 32 filters, each applied to the sketch. Each filter produces a 'feature map' — one of 32 small images you see below. The brighter areas highlight where I am detecting simple features, such as sharp lines in the sketch."
    },
    "max_pooling2d": {
        "step": "Pooling Layer 1",
        "description": "Below is a pooling layer that reduces the image size by focusing on the most important features. The lighter areas represent the key parts of the sketch, simplifying the image while preserving the most relevant features."
    },
    "conv2d_1": {
        "step": "Convolutional Layer 2",
        "description": "This is the second convolutional layer, which looks for more complex features in the sketch, building upon the simple patterns identified by the first layer. Here, the 32 feature maps from the previous layer are stacked on top of each other, and another 64 filters are applied. (Mind-blowing, isn’t it?) You'll see areas where I detect shapes that form parts of the object in the sketch."
    },
    "max_pooling2d_1": {
        "step": "Pooling Layer 2",
        "description": "This is another pooling layer that further reduces the image size, allowing me to focus even more on the key features. The lighter areas represent the most important regions of the sketch that the pooling layer retains, simplifying the rest of the image."
    },
    "conv2d_2": {
        "step": "Convolutional Layer 3",
        "description": "This is the third convolutional layer, with 128 filters (applied to the stacked 64 images from the previous layer). It looks for even more advanced patterns based on what I have already found in earlier layers. This is the final filtering step before I convert the information into a different format to make decisions in the next layer."
    },
    "max_pooling2d_2": {
        "step": "Pooling Layer 3",
        "description": "This is another pooling layer that further reduces the image size, allowing me to focus even more on the key features. The lighter areas represent the most important regions of the sketch that the pooling layer retains, simplifying the rest of the image."
    },
    "flatten": {
        "step": "Flatten Layer",
        "description": "This layer flattens the 2D features into a simple 1D list of numbers, which I can use for decision-making in the next layers."
    },
    "dense": {
        "step": "Dense Layer 1",
        "description": "This is it — this is 'what makes an apple an apple, or a pear a pear.' :) The fully connected (dense) layer links all the features together. Based on the patterns I have identified in the previous layers, I start to combine the information to make predictions about what the sketch represents. The brighter areas show where I am focusing on specific features I am trained to recognize."
    },
    "dropout": {
        "step": "Dropout Layer",
        "description": "This layer reduces overfitting."
    },
    "dense_1": {
        "step": "Dense Layer 2",
        "description": "In the final dense layer, I make my decision: apple (left) / pear (right)."
    }
}
