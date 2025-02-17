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
        "description": "In this fully connected dense neural layer, each neuron (line) corresponds to a particular feature of the sketch. These features—like shape, curves, or symmetry—are what I’ve learned to recognize when distinguishing between apples and pears. The brightness of the neurons represents how much each feature has been detected in the sketch. In the next step, I’ll use these learned features to analyze your sketch further, focusing on whether it’s more 'applish' or 'pearish.'"
    },
    "dropout": {
        "step": "Dropout Layer",
        "description": "This layer reduces overfitting."
    },
    "dense_1": {
        "step": "Dense Layer 2",
        "description": "In the final layer, I evaluate my decision by combining the information from all the features detected in the previous layers. The output from the previous layer helps determine the strength of each feature. Each neuron’s output is influenced by both its activation (how much the feature is detected in the sketch, represented by its brightness in the plot) and the weight (how important that feature is for the classification). For each class (apple or pear), the model computes a score by multiplying the activation of each feature (from the previous layer) by the weight for that class and summing the results. The stronger the activation and the higher the weight of a particular feature for a given class, the higher the score for that class. After evaluating the scores for both classes, the class with the higher score determines my final prediction: apple (left) or pear (right)."
    }
}
