layer_info = {
    "conv2d": {
        "step": "Convolutional Layer 1",
        "description": "This is the first convolutional layer, which scans the input sketch with small filters to detect basic features. In this case, there are 32 filters, each applied to the sketch. Each filter produces a 'feature map' — one of 32 small images you see below. The brighter areas highlight where I am detecting simple features, such as sharp lines in the sketch."
    },
    "max_pooling2d": {
        "step": "Pooling Layer 1",
        "description": "Below is a pooling layer that reduces the image size by focusing on the most important features."
    },
    "conv2d_1": {
        "step": "Convolutional Layer 2",
        "description": "This is the second convolutional layer, which looks for more complex features in the sketch, building on the simple patterns identified by the first layer. The 32 feature maps from the previous layer serve as input, and 64 new filters are applied to generate 64 feature maps. (Pretty mind-blowing, right?)"
    },
    "max_pooling2d_1": {
        "step": "Pooling Layer 2",
        "description": "This is another pooling layer that further reduces the image size, allowing me to focus even more on the key features."
    },
    "conv2d_2": {
        "step": "Convolutional Layer 3",
        "description": "This is the third convolutional layer, which applies 128 filters to the 64 feature maps from the previous layer. It looks for even more advanced patterns based on the features already detected in earlier layers. This is the final filtering step before the information is transformed into a different format for decision-making in the following layers."
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
        "description": """
        In this fully connected dense neural layer, each neuron (line) corresponds to a particular filtered high level feature -like shape, curves, or symmetry- of the sketch. 
        \n
        There are 128 neurons in the picture, from which only a few are activated (bright).
        Again, the brightness of the neurons represents how much each feature has been detected in the sketch. 
        \n
        And now comes the fun part.
        """
    },
    "dropout": {
        "step": "Dropout Layer",
        "description": "This layer reduces overfitting."
    },
    "dense_1": {
        "step": "Dense Layer 2",
        "description": """
        
        
        """
    }
}
