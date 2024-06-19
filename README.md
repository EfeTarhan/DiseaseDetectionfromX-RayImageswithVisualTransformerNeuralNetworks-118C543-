<h1 align="center">Disease Detection from X-Ray Images with Classical Backbone Networks</h1>

<p align="center">
  <img src="https://github.com/tarhanefe/bilkent-cs115-labs/assets/73281981/353e59fa-4cf5-4be5-b62f-afa383f3fdcd" alt="Bilkent University Logo" width = "300" />
</p>

This TUBITAK project aims to develop a system for detecting diseases from X-ray images using neural networks. It involves preparing and labeling a dataset, preprocessing images to 224x224 pixels, and binarizing labels for classification. Data augmentation techniques like rotation, zoom, and shifts enhance the training dataset.

<p align="center">
<img src="https://github.com/tarhanefe/tarhanefe.github.io/assets/73281981/e7b0ee8e-a33d-463a-be0e-b4b79c713190" alt="Project Image" width = "400" /></p>



The core model used is VGG19, a convolutional neural network pre-trained on ImageNet, which serves as the backbone. This model is followed by a series of layers, including flattening and dropout, with a final dense layer that uses a softmax activation function to classify the images into multiple disease categories. The model is trained using the augmented data, with specific parameters for batch size and epochs, aiming to improve diagnostic accuracy and efficiency in detecting diseases from X-ray images.


[Link to the Github Repository](https://github.com/tarhanefe/DiseaseDetectionfromX-RayImageswithVisualTransformerNeuralNetworks-118C543-)

